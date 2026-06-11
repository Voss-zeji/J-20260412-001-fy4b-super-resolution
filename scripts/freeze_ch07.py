#!/usr/bin/env python3
"""
freeze_ch07.py - CH07 模型冻结 + 全圆盘产品生成

任务:
1. 训练并保存 3 个 CH07 代表模型的权重:
   - SFG-SwinIR (主线模型)
   - EmambaIR (精度上限对照)
   - Physics-PFTSR (物理约束参考)

2. 生成 CH07 全圆盘 2km 超分产品:
   - CH07_SR_2km_full_disk
   - CH07_upsample_2km_full_disk
   - CH07_highfreq = CH07_SR_2km - CH07_upsample_2km

执行方式:
    nohup /root/miniconda3/envs/mamba2/bin/python freeze_ch07.py > freeze_ch07.log 2>&1 &
"""

import os, sys, json, time, math, argparse
from pathlib import Path
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
PRODUCT_DIR = Path('/root/autodl-tmp/products/ch07')
DATA_4KM = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH07')
DATA_2KM = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 归一化参数
NORM_MIN = 150.0
NORM_MAX = 350.0


def normalize(img):
    """归一化到 [-1, 1]"""
    img = (img - NORM_MIN) / (NORM_MAX - NORM_MIN)
    return img * 2 - 1


def denormalize(tensor):
    """反归一化"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    tensor = (tensor + 1) / 2.0
    tensor = tensor * (NORM_MAX - NORM_MIN) + NORM_MIN
    return tensor


# =============================================================================
# 模型定义: SFG-SwinIR (主线模型, method 31)
# =============================================================================
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim; self.window_size = window_size; self.num_heads = num_heads
        head_dim = dim // num_heads; self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop); self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords_h = torch.arange(window_size); coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1; relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale; attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None: attn = attn + mask
        attn = F.softmax(attn, dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim; self.num_heads = num_heads; self.window_size = window_size; self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim); self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim); mlp_hidden_dim = int(dim * mlp_ratio); self.mlp = MLP(dim, mlp_hidden_dim, drop)

    def forward(self, x, H, W):
        B, N, C = x.shape; shortcut = x; x = self.norm1(x); x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = shifted_x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = attn_windows.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, -1)
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :H, :W, :].contiguous().view(B, H * W, C)
        x = shortcut + x; x = x + self.mlp(self.norm2(x))
        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop)
            for i in range(depth)])

    def forward(self, x, H, W):
        for blk in self.blocks: x = blk(x, H, W)
        return x


class FrequencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_gen = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.Sigmoid())
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        B, C, H, W = x.shape
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        real, imag = fft_feat.real, fft_feat.imag
        fft_cat = torch.cat([real, imag], dim=1)
        gate = self.gate_gen(fft_cat)
        freq_processed = self.freq_conv(fft_cat)
        freq_gated = freq_processed * gate
        freq_complex = torch.complex(freq_gated, torch.zeros_like(freq_gated))
        spatial = torch.fft.irfft2(freq_complex, s=(H, W), norm='ortho')
        return x + spatial


class SFGSwinIR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, embed_dim=60, depths=[4,4], num_heads=[4,4], window_size=8, mlp_ratio=2.0, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor; self.window_size = window_size
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.num_layers = len(depths); self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(dim=embed_dim, depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio))
        self.norm = nn.LayerNorm(embed_dim); self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.freq_gate = FrequencyGate(embed_dim)
        self.upsample = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') if name != 'conv_last' else nn.init.constant_(m.weight, 0)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_input = x; x_first = self.conv_first(x); B, C, H, W = x_first.shape
        res = x_first; x = x_first.flatten(2).transpose(1, 2)
        for layer in self.layers: x = layer(x, H, W)
        x = self.norm(x); x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv_after_body(x) + res; x = self.freq_gate(x)
        x = self.upsample(x); x = self.conv_last(x)
        base = F.interpolate(x_input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        return x + base


# =============================================================================
# 模型定义: EmambaIR (精度上限, method 17)
# =============================================================================
class SimplifiedMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.proj = nn.Linear(dim, dim * 2)
        self.dt = nn.Linear(dim, dim); self.A = nn.Parameter(torch.randn(dim, dim // 4))
        self.D = nn.Parameter(torch.ones(dim)); self.o = nn.Linear(dim, dim); self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape; x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat); xz = self.proj(x_norm); x_inner, z = xz.chunk(2, dim=-1)
        dt = F.softplus(self.dt(x_inner)); y = x_inner * dt
        y = y * torch.sigmoid(z); y = y + x_flat * self.D
        out = self.o(y); out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class MambaSRBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.GELU(), nn.Conv2d(channels, channels, 3, 1, 1))
        self.mamba = SimplifiedMambaBlock(channels); self.norm = nn.InstanceNorm2d(channels); self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x; out = self.conv(x); out = self.mamba(out); out = self.norm(out)
        return residual + self.gamma * out


class EmambaIRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_blocks=8):
        super().__init__()
        self.shallow = nn.Sequential(nn.Conv2d(in_channels, base_channels, 3, 1, 1), nn.GELU())
        self.mamba_blocks = nn.ModuleList([MambaSRBlock(base_channels) for _ in range(num_blocks)])
        self.global_fusion = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels, 1, 1, 0), nn.GELU())
        self.upsample = nn.Sequential(nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1), nn.PixelShuffle(2), nn.GELU())
        self.reconstruct = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        shallow_feat = self.shallow(x); mamba_feat = shallow_feat
        for block in self.mamba_blocks: mamba_feat = block(mamba_feat)
        fused = self.global_fusion(torch.cat([shallow_feat, mamba_feat], dim=1))
        upsampled = self.upsample(fused); out = self.reconstruct(upsampled)
        bicubic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        return out + bicubic


# =============================================================================
# 模型定义: Physics-PFTSR (物理约束, method 32)
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x): return x + self.conv2(self.relu(self.conv1(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // reduction, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class PFTSR(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feat=64, num_rb=4, upscale=2):
        super().__init__(); self.upscale = upscale
        self.shallow = nn.Sequential(nn.Conv2d(in_ch, feat, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(feat, feat, 3, padding=1))
        self.res_blocks = nn.ModuleList([ResidualBlock(feat) for _ in range(num_rb)]); self.attn = ChannelAttention(feat)
        self.upsample = nn.Sequential(nn.Conv2d(feat, feat * 4, 3, padding=1), nn.PixelShuffle(2))
        self.reconstruct = nn.Conv2d(feat, out_ch, 3, padding=1); nn.init.constant_(self.reconstruct.weight, 0)

    def forward(self, x):
        feat = self.shallow(x)
        for rb in self.res_blocks: feat = rb(feat)
        feat = self.attn(feat); feat = self.upsample(feat); out = self.reconstruct(feat)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return out + base


# =============================================================================
# 训练函数
# =============================================================================
def train_model(model, model_name, epochs=200, batch_size=8, lr=1e-4):
    """训练单个模型并保存最佳 checkpoint"""
    print(f"\n{'='*60}")
    print(f"训练: {model_name}")
    print(f"{'='*60}")

    low_res_dir = "/root/autodl-tmp/Calibration-FY4B/4000M/CH07"
    high_res_dir = "/root/autodl-tmp/Calibration-FY4B/2000M/CH07"

    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel='Channel07',
        batch_size=batch_size, num_workers=4, patch_size=64, upscale_factor=2
    )

    model = model.to(DEVICE)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    best_psnr = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for lr_img, hr_img, _ in train_loader:
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
            optimizer.zero_grad()
            sr = torch.clamp(model(lr_img), -1, 1)
            loss = criterion(sr, hr_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval(); psnr_sum, ssim_sum, n = 0, 0, 0
        with torch.no_grad():
            for lr_img, hr_img, _ in val_loader:
                lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
                sr = torch.clamp(model(lr_img), -1, 1)
                for i in range(sr.size(0)):
                    psnr_sum += calculate_psnr(sr[i:i+1], hr_img[i:i+1])
                    ssim_sum += calculate_ssim(sr[i:i+1], hr_img[i:i+1])
                n += sr.size(0)

        val_psnr = psnr_sum / n; val_ssim = ssim_sum / n
        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr; best_ssim = val_ssim; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | Best: {best_psnr:.2f}@{best_epoch}")

    # 保存最佳权重
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_CH07_best.pth"
    if best_state:
        torch.save(best_state, ckpt_path)
        print(f"\n✅ 已保存最佳权重: {ckpt_path} (PSNR={best_psnr:.2f} @ epoch {best_epoch})")
    else:
        torch.save(model.state_dict(), ckpt_path)
        print(f"\n⚠️ 已保存最终权重: {ckpt_path}")

    result = {
        "model": model_name, "best_psnr": round(best_psnr, 4),
        "best_ssim": round(best_ssim, 4), "best_epoch": best_epoch,
        "params": sum(p.numel() for p in model.parameters()),
        "status": "success"
    }
    with open(CHECKPOINT_DIR / f"{model_name}_training_result.json", 'w') as f:
        json.dump(result, f, indent=2)

    return model, ckpt_path


# =============================================================================
# 全圆盘推理
# =============================================================================
def infer_full_disk(model, hdf_path, device):
    """对单个 HDF 全圆盘图像进行超分推理"""
    with h5py.File(hdf_path, 'r') as f:
        band_key = list(f.keys())[0]
        img = f[band_key][:].astype(np.float32)

    # 处理 NaN
    nan_mask = np.isnan(img)
    if nan_mask.any():
        valid_mean = np.nanmean(img)
        img = np.where(nan_mask, valid_mean, img)

    # 归一化
    img_norm = normalize(img)

    # 转 tensor [1, 1, H, W]
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    # 推理 (tile-based 避免显存溢出)
    H, W = tensor.shape[2], tensor.shape[3]
    tile_size = 512
    pad = 8
    sr = torch.zeros(1, 1, H * 2, W * 2, device=device)

    model.eval()
    with torch.no_grad():
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H); x_end = min(x + tile_size, W)
                tile = tensor[:, :, y:y_end, x:x_end]
                tile_pad = F.pad(tile, (pad, pad, pad, pad), mode='reflect')
                tile_sr = model(tile_pad)
                tile_sr = tile_sr[:, :, pad:-pad, pad:-pad] if pad > 0 else tile_sr
                sr[:, :, y*2:y_end*2, x*2:x_end*2] = tile_sr[:, :, :(y_end-y)*2, :(x_end-x)*2]

    sr = torch.clamp(sr, -1, 1)
    sr_np = sr.squeeze().cpu().numpy()
    return sr_np, img  # 返回超分结果(归一化)和原始图像(物理值)


def generate_ch07_products(model, model_name, ckpt_path, num_samples=10):
    """生成 CH07 全圆盘产品"""
    print(f"\n{'='*60}")
    print(f"生成产品: {model_name}")
    print(f"{'='*60}")

    # 加载权重
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()

    # 获取文件列表
    files_4km = sorted([f for f in os.listdir(DATA_4KM) if f.endswith('.HDF')])
    files_2km = sorted([f for f in os.listdir(DATA_2KM) if f.endswith('.HDF')])

    # 取前 num_samples 个同时存在 4km 和 2km 的文件
    common_files = []
    for f in files_4km:
        if f in files_2km:
            common_files.append(f)
        if len(common_files) >= num_samples:
            break

    product_subdir = PRODUCT_DIR / model_name
    product_subdir.mkdir(parents=True, exist_ok=True)

    results = []
    for fname in common_files:
        t0 = time.time()
        path_4km = DATA_4KM / fname
        path_2km = DATA_2KM / fname

        # 推理
        sr_norm, img_4km = infer_full_disk(model, path_4km, DEVICE)

        # Bicubic 上采样对照
        img_4km_norm = normalize(img_4km)
        img_4km_tensor = torch.from_numpy(img_4km_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            bicubic_tensor = F.interpolate(img_4km_tensor, scale_factor=2, mode='bicubic', align_corners=False)
        bicubic_norm = bicubic_tensor.squeeze().cpu().numpy()

        # 高频信息
        highfreq_norm = sr_norm - bicubic_norm

        # 反归一化到物理值
        sr_phys = denormalize(sr_norm)
        bicubic_phys = denormalize(bicubic_norm)
        highfreq_phys = sr_phys - bicubic_phys

        # 读取真值 2km (用于验证)
        with h5py.File(path_2km, 'r') as f:
            band_key = list(f.keys())[0]
            hr_img = f[band_key][:].astype(np.float32)
            nan_mask = np.isnan(hr_img)
            if nan_mask.any():
                hr_img = np.where(nan_mask, np.nanmean(hr_img), hr_img)

        # 验证 PSNR (需要裁剪到相同大小)
        h, w = hr_img.shape
        sr_crop = sr_phys[:h, :w]
        hr_norm = normalize(hr_img)
        sr_crop_norm = normalize(sr_crop)
        psnr = calculate_psnr(torch.from_numpy(sr_crop_norm).unsqueeze(0).unsqueeze(0),
                               torch.from_numpy(hr_norm).unsqueeze(0).unsqueeze(0))

        # 保存产品
        base_name = fname.replace('.HDF', '')
        np.save(product_subdir / f"{base_name}_SR.npy", sr_phys.astype(np.float32))
        np.save(product_subdir / f"{base_name}_bicubic.npy", bicubic_phys.astype(np.float32))
        np.save(product_subdir / f"{base_name}_highfreq.npy", highfreq_phys.astype(np.float32))

        results.append({
            "file": fname, "psnr": round(float(psnr), 2),
            "shape_sr": sr_phys.shape, "shape_hr": hr_img.shape,
            "time_sec": round(time.time() - t0, 2)
        })
        print(f"  {fname}: PSNR={psnr:.2f} dB, time={time.time()-t0:.1f}s")

    # 保存汇总
    with open(product_subdir / "product_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    avg_psnr = sum(r["psnr"] for r in results) / len(results)
    print(f"\n✅ {model_name} 产品生成完成, 平均 PSNR={avg_psnr:.2f} dB")
    return results


# =============================================================================
# 主流程
# =============================================================================
def main():
    print("=" * 70)
    print("FY-4B CH07 模型冻结 + 全圆盘产品生成")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"Checkpoint 目录: {CHECKPOINT_DIR}")
    print(f"产品目录: {PRODUCT_DIR}")

    # ---- Phase 1: 训练并保存 3 个模型 ----
    print("\n" + "=" * 70)
    print("Phase 1: 训练模型并保存权重")
    print("=" * 70)

    models = [
        (SFGSwinIR(in_channels=1, out_channels=1, embed_dim=60, depths=[4,4], num_heads=[4,4], window_size=8, mlp_ratio=2.0, upscale_factor=2), "SFGSwinIR", 200, 8, 1e-4),
        (EmambaIRNet(in_channels=1, out_channels=1, base_channels=64, num_blocks=8), "EmambaIR", 200, 16, 2e-4),
        (PFTSR(in_ch=1, out_ch=1, feat=64, num_rb=4, upscale=2), "PhysicsPFTSR", 200, 8, 1e-4),
    ]

    ckpts = {}
    for model, name, epochs, bs, lr in models:
        _, ckpt_path = train_model(model, name, epochs=epochs, batch_size=bs, lr=lr)
        ckpts[name] = ckpt_path

    # ---- Phase 2: 生成全圆盘产品 ----
    print("\n" + "=" * 70)
    print("Phase 2: 生成 CH07 全圆盘产品 (每个模型 10 个样本)")
    print("=" * 70)

    for model, name, _, _, _ in models:
        if name in ckpts:
            generate_ch07_products(model, name, ckpts[name], num_samples=10)

    print("\n" + "=" * 70)
    print("全部完成!")
    print("=" * 70)
    print(f"权重: {CHECKPOINT_DIR}")
    print(f"产品: {PRODUCT_DIR}")


if __name__ == '__main__':
    main()
