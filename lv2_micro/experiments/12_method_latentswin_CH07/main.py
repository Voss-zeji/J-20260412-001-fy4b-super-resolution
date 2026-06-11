#!/usr/bin/env python3
"""
12_method_latentswin - Latent SwinIR

融合路径：LCMSR (潜空间编码/解码架构) + SwinIR (移位窗口注意力)

核心改动（2处以上）：
1. 网络结构：将昂贵的 Swin Transformer 从 64x64 图像空间转移到 16x16 潜空间
   （序列长度从 4096 降至 256），实现注意力计算的显著降本，同时保留长程建模能力。
2. 模型设计：在编码器-解码器架构中引入跨尺度跳跃连接（Cross-Scale Skip Connections），
   将编码器多级下采样特征通过 1x1 卷积投影后直接传递到解码器对应上采样层级，
   补偿潜空间压缩带来的细节损失。
3. 模型设计：增加潜空间门控融合模块（Gated Fusion），在 Swin Transformer 输出后
   使用可学习门控动态控制残差特征与 Transformer 特征的混合比例。

数据形态适配：
- 输入: [B, 1, 64, 64]
- 输出: [B, 1, 128, 128]
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
    parser = argparse.ArgumentParser(description='Latent SwinIR')
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
    def __init__(self, dim, window_size=4, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
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
    def __init__(self, dim, num_heads, window_size=4, shift_size=0,
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
    def __init__(self, dim, depth, num_heads, window_size=4, mlp_ratio=4.0,
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


class LatentEncoder(nn.Module):
    """多级潜空间编码器，带跳跃连接特征"""
    def __init__(self, in_channels=1):
        super().__init__()
        # 64x64 -> 64x64
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 64x64 -> 32x32
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )
        # 32x32 -> 16x16
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        # 16x16 -> latent_dim
        self.enc4 = nn.Conv2d(256, 60, 3, padding=1)

    def forward(self, x):
        f1 = self.enc1(x)   # [B, 64, 64, 64]
        f2 = self.enc2(f1)  # [B, 128, 32, 32]
        f3 = self.enc3(f2)  # [B, 256, 16, 16]
        z = self.enc4(f3)   # [B, 60, 16, 16]
        return z, (f1, f2, f3)


class LatentDecoder(nn.Module):
    """多级潜空间解码器，接收跳跃连接"""
    def __init__(self, latent_dim=60, out_channels=1):
        super().__init__()
        # 16x16 processing
        self.dec1 = nn.Sequential(
            nn.Conv2d(latent_dim + 256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )

        # 16x16 -> 32x32
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )

        # 32x32 processing + skip
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )

        # 32x32 -> 64x64
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        # 64x64 processing + skip
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        # 64x64 -> 128x128
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        # 输出
        self.out_conv = nn.Conv2d(64, out_channels, 3, padding=1)
        nn.init.constant_(self.out_conv.weight, 0)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, z, skips):
        f1, f2, f3 = skips

        # 16x16
        z = torch.cat([z, f3], dim=1)
        z = self.dec1(z)

        # 16 -> 32
        z = self.up1(z)
        z = torch.cat([z, f2], dim=1)
        z = self.dec2(z)

        # 32 -> 64
        z = self.up2(z)
        z = torch.cat([z, f1], dim=1)
        z = self.dec3(z)

        # 64 -> 128
        z = self.up3(z)
        z = self.out_conv(z)
        return z


class GatedFusion(nn.Module):
    """门控融合：动态控制残差与 Transformer 特征的混合"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, res_feat, trans_feat):
        concat = torch.cat([res_feat, trans_feat], dim=1)
        g = self.gate(concat)
        return g * trans_feat + (1 - g) * res_feat


class LatentSwinIR(nn.Module):
    """
    Latent SwinIR: 潜空间 Swin Transformer + 跨尺度跳跃连接 + 门控融合
    """
    def __init__(self, in_channels=1, out_channels=1, latent_dim=60,
                 depths=[4, 4], num_heads=[4, 4], window_size=4,
                 mlp_ratio=2.0, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.encoder = LatentEncoder(in_channels)
        self.decoder = LatentDecoder(latent_dim, out_channels)

        # 潜空间 Swin Transformer
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            self.layers.append(BasicLayer(
                dim=latent_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio
            ))

        self.norm = nn.LayerNorm(latent_dim)
        self.conv_after_body = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)

        # 门控融合
        self.gated_fusion = GatedFusion(latent_dim)

        # 上采样基线
        self.up_baseline = nn.Upsample(
            scale_factor=upscale_factor,
            mode='bicubic',
            align_corners=False
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if 'out_conv' in name:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 编码到潜空间
        z, skips = self.encoder(x)
        _, Cz, Hz, Wz = z.shape
        z_res = z

        # 潜空间 Swin Transformer
        z_seq = z.flatten(2).transpose(1, 2)
        for layer in self.layers:
            z_seq = layer(z_seq, Hz, Wz)
        z_seq = self.norm(z_seq)
        z = z_seq.transpose(1, 2).view(B, Cz, Hz, Wz)

        # 残差连接 + 门控融合
        z = self.conv_after_body(z)
        z = self.gated_fusion(z_res, z)

        # 解码到图像空间
        sr_residual = self.decoder(z, skips)

        # 全局残差
        base = self.up_baseline(x)
        if sr_residual.shape != base.shape:
            sr_residual = F.interpolate(
                sr_residual,
                size=base.shape[2:],
                mode='bicubic',
                align_corners=False
            )
        return sr_residual + base


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for lr, hr, _ in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        if sr.shape != hr.shape:
            sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
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
            if sr.shape != hr.shape:
                sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)
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
    print("12_method_latentswin - Latent SwinIR")
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

    model = LatentSwinIR(
        in_channels=1, out_channels=1, latent_dim=60,
        depths=[4, 4], num_heads=[4, 4], window_size=4,
        mlp_ratio=2.0, upscale_factor=2
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)
    print(f"参数量: {model_params:,} ({model_size_mb:.2f} MB)")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = time.time()
    best_psnr = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device)
            best_psnr = max(best_psnr, metrics['psnr'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{args.epochs}] loss: {train_loss:.4f} | "
                  f"val_psnr: {metrics['psnr']:.2f} dB | val_ssim: {metrics['ssim']:.4f} | "
                  f"lr: {current_lr:.6f}")

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
        "method": "12_method_latentswin",
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
