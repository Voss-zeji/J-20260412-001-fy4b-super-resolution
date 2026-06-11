#!/usr/bin/env python3
"""
31_method_sfg_swinir - Spatial-Frequency Gated SwinIR
基于 SFG-SwinIR 思想：
- 在 SwinIR 深层特征后添加频域门控模块
- FFT 提取频域特征，可学习门控选择重要频率
- 逆 FFT 转回后与空间特征融合
"""

import argparse, json, sys, time, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='SFG-SwinIR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
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
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop)
            for i in range(depth)])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class FrequencyGate(nn.Module):
    """频域门控模块"""
    def __init__(self, channels):
        super().__init__()
        # 门控生成器：基于频域特征生成门控权重
        self.gate_gen = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid()
        )
        # 频域处理
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FFT
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        real = fft_feat.real
        imag = fft_feat.imag
        # 拼接实部和虚部
        fft_cat = torch.cat([real, imag], dim=1)
        # 生成门控
        gate = self.gate_gen(fft_cat)
        # 处理频域特征
        freq_processed = self.freq_conv(fft_cat)
        # 应用门控
        freq_gated = freq_processed * gate
        # IFFT
        freq_complex = torch.complex(freq_gated, torch.zeros_like(freq_gated))
        spatial = torch.fft.irfft2(freq_complex, s=(H, W), norm='ortho')
        # 残差融合
        return x + spatial


class SFGSwinIR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, embed_dim=60, depths=[4, 4],
                 num_heads=[4, 4], window_size=8, mlp_ratio=2.0, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim, depth=depths[i_layer],
                num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        # 频域门控
        self.freq_gate = FrequencyGate(embed_dim)
        if upscale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2))
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * (upscale_factor ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale_factor))
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if name == 'conv_last':
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_input = x
        x_first = self.conv_first(x)
        res = x_first
        B, C, H, W = x_first.shape
        x = x_first.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, H, W)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv_after_body(x) + res
        # 频域门控
        x = self.freq_gate(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        base = F.interpolate(x_input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        return x + base


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = torch.clamp(model(lr), -1, 1)
        loss = criterion(sr, hr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    psnr, ssim, rmse, n = 0, 0, 0, 0
    with torch.no_grad():
        for lr, hr, _ in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = torch.clamp(model(lr), -1, 1)
            for i in range(sr.size(0)):
                psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
                ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
                rmse += calculate_rmse(sr[i:i+1], hr[i:i+1])
            n += sr.size(0)
    return {'psnr': psnr/n, 'ssim': ssim/n, 'rmse': rmse/n}


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    channel = args.band.replace('CH', 'Channel')

    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=4, patch_size=64, upscale_factor=2
    )

    model = SFGSwinIR(in_channels=1, out_channels=1, embed_dim=60, depths=[4, 4],
                      num_heads=[4, 4], window_size=8, mlp_ratio=2.0, upscale_factor=2).to(device)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"[SFGSwinIR] Params: {model_params:,}")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_psnr, best_ssim, best_epoch = 0, 0, 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            best_ssim = metrics['ssim']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | "
                  f"Val PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | Best: {best_psnr:.2f}@{best_epoch}")

    if best_state:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, device)

    dummy = torch.randn(1, 1, 64, 64).to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad(): _ = model(dummy)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    infer_ms = (time.time() - t0) * 1000

    result = {
        "method": "31_method_sfg_swinir",
        "band": args.band,
        "best_psnr": round(best_psnr, 4),
        "best_ssim": round(best_ssim, 4),
        "best_epoch": best_epoch,
        "final_psnr": round(final_metrics['psnr'], 4),
        "final_ssim": round(final_metrics['ssim'], 4),
        "train_epochs": args.epochs,
        "model_params": model_params,
        "inference_time_ms": round(infer_ms, 2),
        "train_time": round(time.time() - start_time, 2),
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone! Best PSNR: {best_psnr:.2f} dB @ epoch {best_epoch}")

if __name__ == '__main__':
    main()
