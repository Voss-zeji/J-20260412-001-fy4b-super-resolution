#!/usr/bin/env python3
"""
ch08_data_preparation.py - CH08 数据准备 (阶段 1)

任务:
1. 清理 root 文件系统空间
2. 建立 CH07/CH08 同时刻配对索引
3. 生成 50 个验证时刻的全圆盘产品
4. 生成训练 patch 数据集 (所有配对时刻, 每时刻 16 个随机 patch)

输出目录: /root/autodl-tmp/fy4b_ch08_data/
"""

import os, sys, json, time, random
from pathlib import Path
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 配置
# =============================================================================
DATA_4KM_CH07 = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH07')
DATA_2KM_CH07 = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')
DATA_4KM_CH08 = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH08')
OUTPUT_DIR = Path('/root/autodl-tmp/fy4b_ch08_data')
CKPT_PATH = Path('/root/jobs/J-20260412-001-fy4b-super-resolution/checkpoints/SFGSwinIR_CH07_best.pth')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORM_MIN, NORM_MAX = 150.0, 350.0
PATCH_SIZE_4KM = 64   # 4km 分辨率 patch 尺寸
PATCH_SIZE_2KM = 128  # 2km 分辨率 patch 尺寸 (= 64 * 2)
PATCHES_PER_IMAGE = 16
VAL_SAMPLE_COUNT = 50


def normalize(img):
    img = (img - NORM_MIN) / (NORM_MAX - NORM_MIN)
    return img * 2 - 1


def denormalize(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    tensor = (tensor + 1) / 2.0
    tensor = tensor * (NORM_MAX - NORM_MIN) + NORM_MIN
    return tensor


# =============================================================================
# 模型定义: SFG-SwinIR
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
# 数据读取
# =============================================================================
def read_hdf(path):
    with h5py.File(path, 'r') as f:
        key = list(f.keys())[0]
        img = f[key][:].astype(np.float32)
    nan_mask = np.isnan(img)
    if nan_mask.any():
        img = np.where(nan_mask, np.nanmean(img), img)
    return img


def infer_ch07_sr(model, img_4km, device):
    """对单张 4km 全圆盘图像做 SR 推理"""
    img_norm = normalize(img_4km)
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    H, W = tensor.shape[2], tensor.shape[3]
    tile_size = 512; pad = 8
    sr = torch.zeros(1, 1, H * 2, W * 2, device=device)
    model.eval()
    with torch.no_grad():
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H); x_end = min(x + tile_size, W)
                tile = tensor[:, :, y:y_end, x:x_end]
                tile_pad = F.pad(tile, (pad, pad, pad, pad), mode='reflect')
                tile_sr = model(tile_pad)
                tile_sr = tile_sr[:, :, pad:-pad, pad:-pad]
                sr[:, :, y*2:y_end*2, x*2:x_end*2] = tile_sr[:, :, :(y_end-y)*2, :(x_end-x)*2]
    sr = torch.clamp(sr, -1, 1)
    sr_np = sr.squeeze().cpu().numpy()
    bic = F.interpolate(tensor, scale_factor=2, mode='bicubic', align_corners=False)
    bic_np = bic.squeeze().cpu().numpy()
    highfreq = sr_np - bic_np
    return denormalize(sr_np), denormalize(bic_np), denormalize(highfreq)


# =============================================================================
# 阶段 1: 清理 root + 建立配对索引
# =============================================================================
def step1_clean_and_index():
    print("=" * 60)
    print("Step 1: 清理 root + 建立配对索引")
    print("=" * 60)

    # 清理 root
    os.system('> /var/log/btmp 2>/dev/null')
    os.system('rm -rf /tmp/mu_* /tmp/test_* /tmp/torchinductor_root /tmp/mineru-* 2>/dev/null')
    print("  root 清理完成")

    # 扫描文件
    def get_timestamps(dir_path):
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.HDF')])
        ts_map = {}
        for f in files:
            ts = f.split('_')[-1].replace('.HDF', '')
            ts_map[ts] = f
        return ts_map

    ch07_4km = get_timestamps(DATA_4KM_CH07)
    ch07_2km = get_timestamps(DATA_2KM_CH07)
    ch08_4km = get_timestamps(DATA_4KM_CH08)

    common = sorted(set(ch07_4km.keys()) & set(ch08_4km.keys()))
    print(f"  CH07 4km: {len(ch07_4km)} 文件")
    print(f"  CH08 4km: {len(ch08_4km)} 文件")
    print(f"  CH07 2km: {len(ch07_2km)} 文件")
    print(f"  配对时刻: {len(common)}")

    pairs = []
    for ts in common:
        p = {"timestamp": ts, "ch07_4km": str(DATA_4KM_CH07 / ch07_4km[ts]), "ch08_4km": str(DATA_4KM_CH08 / ch08_4km[ts]), "has_ch07_2km": ts in ch07_2km}
        if ts in ch07_2km:
            p["ch07_2km"] = str(DATA_2KM_CH07 / ch07_2km[ts])
        pairs.append(p)

    # 划分: 80% 训练, 10% 验证, 10% 测试
    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    for i, p in enumerate(pairs):
        if i < train_end: p["split"] = "train"
        elif i < val_end: p["split"] = "val"
        else: p["split"] = "test"

    index_path = OUTPUT_DIR / "pair_index.json"
    with open(index_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"  划分: train={train_end}, val={val_end-train_end}, test={n-val_end}")
    print(f"  索引已保存: {index_path}")
    return pairs


# =============================================================================
# 阶段 2: 生成验证集全圆盘产品
# =============================================================================
def step2_validation_full_disk(pairs, model):
    print("\n" + "=" * 60)
    print("Step 2: 生成验证集全圆盘产品")
    print("=" * 60)

    val_pairs = [p for p in pairs if p["split"] == "val"][:VAL_SAMPLE_COUNT]
    val_dir = OUTPUT_DIR / "validation_full_disk"
    val_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(val_pairs):
        ts = p["timestamp"]
        subdir = val_dir / ts
        subdir.mkdir(parents=True, exist_ok=True)

        # CH07 4km
        ch07_4km = read_hdf(p["ch07_4km"])
        np.save(subdir / "CH07_4km.npy", ch07_4km.astype(np.float32))

        # CH08 4km
        ch08_4km = read_hdf(p["ch08_4km"])
        np.save(subdir / "CH08_4km.npy", ch08_4km.astype(np.float32))

        # CH07 SR + highfreq
        sr, bic, hf = infer_ch07_sr(model, ch07_4km, DEVICE)
        np.save(subdir / "CH07_SR_2km.npy", sr.astype(np.float32))
        np.save(subdir / "CH07_bicubic_2km.npy", bic.astype(np.float32))
        np.save(subdir / "CH07_highfreq.npy", hf.astype(np.float32))

        # CH08 upsample
        ch08_norm = normalize(ch08_4km)
        ch08_t = torch.from_numpy(ch08_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            ch08_bic = F.interpolate(ch08_t, scale_factor=2, mode='bicubic', align_corners=False)
        ch08_bic = torch.clamp(ch08_bic, -1, 1)
        ch08_bic_np = denormalize(ch08_bic.squeeze().cpu().numpy())
        np.save(subdir / "CH08_upsample_2km.npy", ch08_bic_np.astype(np.float32))

        # CH07 2km true (如果存在)
        if p.get("ch07_2km"):
            ch07_2km = read_hdf(p["ch07_2km"])
            np.save(subdir / "CH07_2km_true.npy", ch07_2km.astype(np.float32))

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(val_pairs)}")

    print(f"  验证集产品已保存: {val_dir}")


# =============================================================================
# 阶段 3: 生成训练 patch 数据集
# =============================================================================
def step3_train_patches(pairs, model):
    print("\n" + "=" * 60)
    print("Step 3: 生成训练 patch 数据集")
    print("=" * 60)

    train_pairs = [p for p in pairs if p["split"] == "train"]
    patch_dir = OUTPUT_DIR / "train_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    patch_records = []
    patch_idx = 0

    for pair_idx, p in enumerate(train_pairs):
        ts = p["timestamp"]

        # 读取数据
        ch07_4km = read_hdf(p["ch07_4km"])
        ch08_4km = read_hdf(p["ch08_4km"])

        # CH07 SR
        sr, bic, hf = infer_ch07_sr(model, ch07_4km, DEVICE)

        H, W = ch07_4km.shape
        max_y = H - PATCH_SIZE_4KM
        max_x = W - PATCH_SIZE_4KM

        # 随机采样 patch
        rng = random.Random(int(ts))  # 固定随机种子保证可复现
        positions = []
        for _ in range(PATCHES_PER_IMAGE):
            y = rng.randint(0, max_y)
            x = rng.randint(0, max_x)
            positions.append((y, x))

        for y, x in positions:
            # 4km patch
            ch07_patch = ch07_4km[y:y+PATCH_SIZE_4KM, x:x+PATCH_SIZE_4KM]
            ch08_patch = ch08_4km[y:y+PATCH_SIZE_4KM, x:x+PATCH_SIZE_4KM]

            # 2km patch
            y2, x2 = y * 2, x * 2
            sr_patch = sr[y2:y2+PATCH_SIZE_2KM, x2:x2+PATCH_SIZE_2KM]
            hf_patch = hf[y2:y2+PATCH_SIZE_2KM, x2:x2+PATCH_SIZE_2KM]
            bic_patch = bic[y2:y2+PATCH_SIZE_2KM, x2:x2+PATCH_SIZE_2KM]

            # CH08 upsample patch
            ch08_norm = normalize(ch08_patch)
            ch08_t = torch.from_numpy(ch08_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                ch08_up = F.interpolate(ch08_t, scale_factor=2, mode='bicubic', align_corners=False)
            ch08_up = torch.clamp(ch08_up, -1, 1)
            ch08_up_patch = denormalize(ch08_up.squeeze().cpu().numpy())

            # 保存 patch
            patch_file = patch_dir / f"patch_{patch_idx:07d}.npz"
            np.savez_compressed(patch_file,
                ch07_4km=ch07_patch.astype(np.float32),
                ch08_4km=ch08_patch.astype(np.float32),
                ch07_sr=sr_patch.astype(np.float32),
                ch07_highfreq=hf_patch.astype(np.float32),
                ch07_bicubic=bic_patch.astype(np.float32),
                ch08_upsample=ch08_up_patch.astype(np.float32),
                timestamp=ts, y=y, x=x
            )

            patch_records.append({"idx": patch_idx, "timestamp": ts, "y": y, "x": x, "file": str(patch_file)})
            patch_idx += 1

        if (pair_idx + 1) % 50 == 0:
            print(f"  已处理 {pair_idx+1}/{len(train_pairs)} 个时刻, {patch_idx} 个 patches")

    # 保存 patch 索引
    with open(patch_dir / "patch_index.json", 'w') as f:
        json.dump(patch_records, f, indent=2)

    print(f"  训练 patch 已保存: {patch_dir}")
    print(f"  总计: {patch_idx} 个 patches")
    print(f"  预计空间: ~{patch_idx * 0.2 / 1024:.1f} GB")


# =============================================================================
# 主流程
# =============================================================================
def main():
    print("=" * 70)
    print("FY-4B CH08 数据准备")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 加载模型
    print("\n加载 SFG-SwinIR 模型...")
    model = SFGSwinIR(in_channels=1, out_channels=1, embed_dim=60, depths=[4,4], num_heads=[4,4], window_size=8, mlp_ratio=2.0, upscale_factor=2)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 执行阶段
    pairs = step1_clean_and_index()
    step2_validation_full_disk(pairs, model)
    step3_train_patches(pairs, model)

    print("\n" + "=" * 70)
    print("CH08 数据准备完成!")
    print("=" * 70)
    print(f"输出位置: {OUTPUT_DIR}")
    print(f"  - pair_index.json: 配对索引")
    print(f"  - validation_full_disk/: 验证集全圆盘产品 ({VAL_SAMPLE_COUNT} 个时刻)")
    print(f"  - train_patches/: 训练 patch 数据集")


if __name__ == '__main__':
    main()
