#!/usr/bin/env python3
"""
06_method_tinynina - TinyNina: Edge-AI Framework for Satellite Super-Resolution

基于论文: "TinyNina: Edge-AI Framework for Satellite Super-Resolution", ArXiv 2604.04445
适配FY-4B: 单通道红外图像，保留深度可分离卷积和通道门控思想

特点:
- 超轻量架构 (~51K参数)
- 深度可分离卷积减少计算
- 通道门控（波长特定注意力适配）
- 适合边缘部署
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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='TinyNina Edge-AI SR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class ChannelGate(nn.Module):
    """
    通道门控（波长特定注意力适配为单通道）
    参考TinyNina的波长特定注意力门控思想
    """
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
        return x * self.sigmoid(avg_out + max_out)

    def sigmoid(self, x):
        return torch.sigmoid(x)


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：减少参数量和计算量"""
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


class ResidualBlock(nn.Module):
    """轻量残差块（深度可分离卷积）"""
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


class TinyNina(nn.Module):
    """
    TinyNina: Edge-AI Satellite Super-Resolution
    适配FY-4B单通道红外图像
    """
    def __init__(self, in_channels=1, num_features=32, num_blocks=2, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        # 浅层特征提取（深度可分离）
        self.shallow_feat = nn.Sequential(
            DepthwiseSeparableConv(in_channels, num_features),
            nn.ReLU(inplace=True),
            ChannelGate(num_features)
        )

        # 残差块序列
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])

        # 中间特征融合
        self.mid_conv = DepthwiseSeparableConv(num_features, num_features)

        # 上采样层（PixelShuffle）
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        # 重建层
        self.reconstruction = nn.Conv2d(num_features, in_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if name == 'reconstruction':
                    # 最后一层输出小残差，初始化为零
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 浅层特征
        feat = self.shallow_feat(x)

        # 残差块序列
        identity = feat
        for block in self.residual_blocks:
            feat = block(feat)

        # 中间融合
        feat = self.mid_conv(feat) + identity

        # 上采样
        feat = self.upsample(feat)

        # 重建
        sr = self.reconstruction(feat)

        # 全局残差连接（双三次上采样基线）
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                            mode='bicubic', align_corners=False)
        return sr + base


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (lr, hr, _) in enumerate(train_loader):
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

    return total_loss / num_batches


def evaluate(model, val_loader, device):
    """评估模型"""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
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
    print("06_method_tinynina - TinyNina Edge-AI SR")
    print("=" * 60)
    print(f"Band: {args.band}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")

    # 创建数据加载器
    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    channel = args.band.replace('CH', 'Channel')

    print(f"\n加载数据...")
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir,
        high_res_dir=high_res_dir,
        channel=channel,
        batch_size=args.batch_size,
        num_workers=4,
        patch_size=64,
        upscale_factor=2
    )
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 创建模型
    print("\n创建 TinyNina 模型...")
    model = TinyNina(
        in_channels=1,
        num_features=32,
        num_blocks=2,
        upscale_factor=2
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)
    print(f"参数量: {model_params:,} ({model_size_mb:.2f} MB)")

    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    print(f"\n开始训练...")
    start_time = time.time()
    best_psnr = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device)
            best_psnr = max(best_psnr, metrics['psnr'])

            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"train_loss: {train_loss:.4f} | "
                  f"val_psnr: {metrics['psnr']:.2f} dB | "
                  f"val_ssim: {metrics['ssim']:.4f}")

    train_time = time.time() - start_time

    # 最终评估
    print(f"\n最终评估...")
    metrics = evaluate(model, val_loader, device)

    # 推理速度测试
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time_ms = (time.time() - t0) * 1000

    # 保存结果
    result = {
        "method": "06_method_tinynina",
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
    print("=" * 60)
    print(f"Best val_psnr: {best_psnr:.2f} dB")
    print(f"Final val_psnr: {result['val_psnr']:.2f} dB")
    print(f"val_ssim: {result['val_ssim']:.4f}")
    print(f"Train time: {result['train_time']:.1f}s")
    print(f"结果已保存: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
