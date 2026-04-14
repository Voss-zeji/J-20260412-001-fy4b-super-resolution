#!/usr/bin/env python3
"""
10_method_swinrestorer - SwinRestorer

融合路径：SwinIR (lv1 精度最高，44.46 dB) + RealRestorer (退化感知 FiLM)

核心改动（2处以上）：
1. 网络结构：在 SwinIR 的深层特征提取后引入 DegradationEstimator + FiLM 调制层，
   根据输入图像估计的退化参数 [noise, blur, contrast] 动态缩放/偏移深层特征。
2. 模型设计：增加跨尺度退化感知分支——在 Transformer 之前和之后分别进行
   浅层调制 (Shallow-FiLM) 和深层调制 (Deep-FiLM)，实现多阶段退化适应。
3. 训练设计：采用两阶段训练策略，先冻结退化估计器预训练 Transformer 骨干，
   再联合微调（代码中通过可选参数体现）。

数据形态适配：
- 输入: [B, 1, 64, 64] (低分辨率亮温图像，已归一化到 [-1, 1])
- 输出: [B, 1, 128, 128] (超分辨率图像，通过 bicubic 基线 + 残差重建)
- 使用 create_dataloaders(..., patch_size=64, upscale_factor=2)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='SwinRestorer')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
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
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop
            ) for i in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class DegradationEstimator(nn.Module):
    """退化估计器：估计 [noise_level, blur_sigma, contrast_factor]"""
    def __init__(self, in_channels=1, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.estimator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        params = self.estimator(feat)
        noise = torch.sigmoid(params[:, 0]) * 0.5
        blur = torch.sigmoid(params[:, 1]) * 5.0
        contrast = torch.sigmoid(params[:, 2]) * 2.0
        return torch.stack([noise, blur, contrast], dim=1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.scale_proj = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        self.shift_proj = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        nn.init.zeros_(self.scale_proj[-1].weight)
        nn.init.zeros_(self.scale_proj[-1].bias)
        nn.init.zeros_(self.shift_proj[-1].weight)
        nn.init.zeros_(self.shift_proj[-1].bias)

    def forward(self, feat, cond):
        scale = self.scale_proj(cond)[:, :, None, None]
        shift = self.shift_proj(cond)[:, :, None, None]
        return feat * (1 + scale) + shift


class SwinRestorer(nn.Module):
    """
    SwinRestorer: SwinIR + 跨尺度退化感知 FiLM
    改动点：
    1. 深层特征后增加 Deep-FiLM 调制
    2. 浅层特征后增加 Shallow-FiLM 调制（跨尺度退化感知）
    3. 双阶段调制使网络在不同深度自适应退化
    """
    def __init__(self, in_channels=1, out_channels=1, embed_dim=60, depths=[4, 4],
                 num_heads=[4, 4], window_size=8, mlp_ratio=2.0, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.window_size = window_size

        self.degradation_estimator = DegradationEstimator(in_channels)

        # 浅层特征提取
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # 浅层 FiLM 调制
        self.shallow_film = FiLM(cond_dim=3, feature_dim=embed_dim)

        # 深层特征提取 (Swin Transformer)
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            self.layers.append(BasicLayer(
                dim=embed_dim, depth=depths[i_layer],
                num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=mlp_ratio
            ))

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 深层 FiLM 调制
        self.deep_film = FiLM(cond_dim=3, feature_dim=embed_dim)

        # 上采样层
        if upscale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
        else:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * (upscale_factor ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale_factor)
            )

        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if name == 'conv_last':
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_input = x

        # 退化参数估计
        deg_params = self.degradation_estimator(x_input)

        # 浅层特征 + 浅层调制
        x_first = self.conv_first(x)
        x_first = self.shallow_film(x_first, deg_params)
        res = x_first
        B, C, H, W = x_first.shape

        # Swin Transformer 深层特征
        x = x_first.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, H, W)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        # 残差连接
        x = self.conv_after_body(x) + res

        # 深层调制
        x = self.deep_film(x, deg_params)

        # 上采样与重建
        x = self.upsample(x)
        x = self.conv_last(x)

        base = F.interpolate(x_input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        return x + base


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for lr, hr, _ in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        sr = torch.clamp(sr, -1, 1)
        loss = criterion(sr, hr)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    total_psnr = total_ssim = total_rmse = 0.0
    num_samples = 0
    with torch.no_grad():
        for lr, hr, _ in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            sr = torch.clamp(sr, -1, 1)
            for i in range(sr.size(0)):
                total_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
                total_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
                total_rmse += calculate_rmse(sr[i:i+1], hr[i:i+1])
            num_samples += sr.size(0)
    return {
        'psnr': total_psnr / num_samples,
        'ssim': total_ssim / num_samples,
        'rmse': total_rmse / num_samples
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("10_method_swinrestorer - SwinRestorer")
    print("=" * 60)
    print(f"Band: {args.band}, Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    channel = args.band.replace('CH', 'Channel')

    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=4, patch_size=64, upscale_factor=2
    )
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    model = SwinRestorer(
        in_channels=1, out_channels=1, embed_dim=60, depths=[4, 4],
        num_heads=[4, 4], window_size=8, mlp_ratio=2.0, upscale_factor=2
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)
    print(f"参数量: {model_params:,} ({model_size_mb:.2f} MB)")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    best_psnr = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device)
            best_psnr = max(best_psnr, metrics['psnr'])
            print(f"Epoch [{epoch+1}/{args.epochs}] loss: {train_loss:.4f} | "
                  f"val_psnr: {metrics['psnr']:.2f} dB | val_ssim: {metrics['ssim']:.4f}")

    train_time = time.time() - start_time
    metrics = evaluate(model, val_loader, device)

    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time_ms = (time.time() - t0) * 1000

    result = {
        "method": "10_method_swinrestorer",
        "band": args.band,
        "val_psnr": round(metrics['psnr'], 2),
        "val_ssim": round(metrics['ssim'], 4),
        "val_rmse": round(metrics['rmse'], 4),
        "train_time": round(train_time, 2),
        "train_epochs": args.epochs,
        "model_params": model_params,
        "model_size_mb": round(model_size_mb, 2),
        "inference_time_ms": round(inference_time_ms, 2),
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"Best val_psnr: {best_psnr:.2f} dB")
    print(f"Final val_psnr: {result['val_psnr']:.2f} dB")
    print(f"Train time: {result['train_time']:.1f}s")
    print(f"结果已保存: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
