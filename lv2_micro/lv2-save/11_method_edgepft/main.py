#!/usr/bin/env python3
"""
11_method_edgepft - Edge-PFT

融合路径：TinyNina (超轻量 Edge-AI，~51K 参数) + PFTSR (渐进特征转移 + CBAM)

核心改动（2处以上）：
1. 网络结构：将 PFTSR 中所有标准 3x3 卷积替换为 TinyNina 的 DepthwiseSeparableConv，
   参数量和计算量降低约 60-70%，同时保留渐进式 PixelShuffle 上采样框架。
2. 模型设计：在 ProgressiveFeatureTransferBlock 中并联 CBAM 通道/空间注意力
   和 TinyNina 的 ChannelGate，通过可学习 1x1 卷积进行自适应注意力融合，
   而非简单的串联或单一注意力。
3. 网络结构：增加多级跳跃连接——将浅层特征通过 1x1 逐点卷积投影到各 PFT 块
   的输出维度，进行逐元素相加，改善梯度流并保留高频细节。

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
    parser = argparse.ArgumentParser(description='Edge-PFT')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelGate(nn.Module):
    """通道门控（来自 TinyNina）"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class LightResidualBlock(nn.Module):
    """轻量残差块：深度可分离卷积 + ChannelGate"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.gate = ChannelGate(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.gate(out)
        return out + identity


class ChannelAttention(nn.Module):
    """CBAM 通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """卷积块注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class LightPFTBlock(nn.Module):
    """
    轻量渐进特征转移块
    改动点：并联 CBAM + ChannelGate，通过可学习 1x1 卷积融合
    """
    def __init__(self, channels, upscale_factor=2, num_rb=2, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        self.residual_blocks = nn.ModuleList([
            LightResidualBlock(channels) for _ in range(num_rb)
        ])

        if use_attention:
            self.cbam = CBAM(channels)
            self.channel_gate = ChannelGate(channels)
            # 可学习注意力融合
            self.attn_fusion = nn.Conv2d(channels * 2, channels, 1, bias=False)

        if upscale_factor == 2:
            self.upsample = nn.Sequential(
                DepthwiseSeparableConv(channels, channels * 4),
                nn.PixelShuffle(2)
            )
        else:
            self.upsample = nn.Identity()

        self.fusion = DepthwiseSeparableConv(channels, channels)

    def forward(self, x):
        identity = x
        feat = x

        for rb in self.residual_blocks:
            feat = rb(feat)

        if self.use_attention:
            feat_cbam = self.cbam(feat)
            feat_gate = self.channel_gate(feat)
            feat = self.attn_fusion(torch.cat([feat_cbam, feat_gate], dim=1))

        feat = feat + identity
        upsampled = self.upsample(feat)
        upsampled = self.fusion(upsampled)
        return upsampled, feat


class EdgePFT(nn.Module):
    """
    Edge-PFT: 轻量渐进特征转移超分辨率网络
    改动点：
    1. 全部标准卷积替换为 DepthwiseSeparableConv
    2. PFT 块内并联 CBAM + ChannelGate 并通过可学习门控融合
    3. 多级跳跃连接：浅层特征投影后逐元素加到各 PFT 块输出
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=48,
                 num_pft_blocks=3, num_rb_per_block=2, upscale_factor=2,
                 use_attention=True):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.num_pft_blocks = num_pft_blocks

        # 浅层特征提取（深度可分离）
        self.shallow_feat = nn.Sequential(
            DepthwiseSeparableConv(in_channels, num_features),
            nn.ReLU(inplace=True),
            ChannelGate(num_features)
        )

        # PFT 块序列
        self.pft_blocks = nn.ModuleList([
            LightPFTBlock(
                num_features,
                upscale_factor=upscale_factor if i == num_pft_blocks - 1 else 1,
                num_rb=num_rb_per_block,
                use_attention=use_attention
            ) for i in range(num_pft_blocks)
        ])

        # 多级跳跃连接投影：将浅层特征投影到每个 PFT 块输出的空间尺寸
        self.skip_projectors = nn.ModuleList()
        for i in range(num_pft_blocks):
            if i == num_pft_blocks - 1:
                # 最后一个块有上采样
                self.skip_projectors.append(
                    nn.Sequential(
                        nn.Conv2d(num_features, num_features * 4, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
            else:
                self.skip_projectors.append(
                    nn.Conv2d(num_features, num_features, 1, bias=False)
                )

        # 全局残差学习
        self.global_residual = nn.Sequential(
            DepthwiseSeparableConv(num_features, num_features),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(num_features, out_channels)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 跳过最后输出层的 pointwise
                if self._is_last_conv(m):
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _is_last_conv(self, m):
        # 判断是否为 global_residual 最后一个 DepthwiseSeparableConv 的 pointwise
        last_block = self.global_residual[-1]
        return m is last_block.pointwise

    def forward(self, x):
        shallow_feat = self.shallow_feat(x)

        feat = shallow_feat
        for i, pft_block in enumerate(self.pft_blocks):
            upsampled_feat, _ = pft_block(feat)
            # 多级跳跃连接
            skip = self.skip_projectors[i](shallow_feat)
            if skip.shape == upsampled_feat.shape:
                upsampled_feat = upsampled_feat + skip
            feat = upsampled_feat

        sr_img = self.global_residual(feat)
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                            mode='bicubic', align_corners=False)
        return sr_img + base


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
    print("11_method_edgepft - Edge-PFT")
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

    model = EdgePFT(
        in_channels=1, out_channels=1, num_features=48,
        num_pft_blocks=3, num_rb_per_block=2, upscale_factor=2,
        use_attention=True
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
        "method": "11_method_edgepft",
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
