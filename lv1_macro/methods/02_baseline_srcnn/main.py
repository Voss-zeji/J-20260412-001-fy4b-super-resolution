#!/usr/bin/env python3
"""
02_baseline_srcnn - SRCNN (Super-Resolution Convolutional Neural Network)

Dong et al., "Learning a Deep Convolutional Network for Image Super-Resolution", ECCV 2014

结构: Conv(9x9) -> ReLU -> Conv(1x1) -> ReLU -> Conv(5x5)
特点: 深度学习超分辨率的先驱，结构简单但有效
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='SRCNN Baseline')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class SRCNN(nn.Module):
    """
    SRCNN 网络结构
    三层卷积: 特征提取 -> 非线性映射 -> 重建
    """
    def __init__(self, num_channels=1, f1=9, f2=1, f3=5, n1=64, n2=32):
        super(SRCNN, self).__init__()
        # 第一层: 特征提取 (patch extraction)
        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=f1, padding=f1//2)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二层: 非线性映射 (non-linear mapping)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2//2)
        self.relu2 = nn.ReLU(inplace=True)

        # 第三层: 重建 (reconstruction)
        self.conv3 = nn.Conv2d(n2, num_channels, kernel_size=f3, padding=f3//2)
        # 初始化为零，使模型从 bicubic base 开始学习
        nn.init.constant_(self.conv3.weight, 0)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0)

    def forward(self, x):
        # 先进行双三次插值上采样
        x = nn.functional.interpolate(
            x, scale_factor=2, mode='bicubic', align_corners=False
        )
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
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
            # 裁剪到有效范围
            sr = torch.clamp(sr, -1, 1)

            # 计算指标
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

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("02_baseline_srcnn - SRCNN")
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
    print("\n创建 SRCNN 模型...")
    model = SRCNN(num_channels=1).to(device)
    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)
    print(f"参数量: {model_params:,} ({model_size_mb:.2f} MB)")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    print(f"\n开始训练...")
    start_time = time.time()
    best_psnr = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 每5个epoch验证一次
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
        "method": "02_baseline_srcnn",
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

    # 确保输出目录存在
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
