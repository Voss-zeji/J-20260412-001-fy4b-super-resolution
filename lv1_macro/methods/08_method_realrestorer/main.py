#!/usr/bin/env python3
"""
08_method_realrestorer - RealRestorer: Real-World Image Restoration

基于论文: "RealRestorer: Real-World Image Restoration", 2026
适配FY-4B: 真实卫星图像退化建模（大气扰动、传感器噪声）

特点:
- 退化估计器：自适应估计噪声/模糊参数
- 条件残差块：根据退化类型调整处理
- 无需配对训练数据
- 适合真实卫星成像条件
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
    parser = argparse.ArgumentParser(description='RealRestorer Real-World SR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class DegradationEstimator(nn.Module):
    """
    退化估计器：估计图像退化参数
    输出: [noise_level, blur_sigma, contrast_factor]
    """
    def __init__(self, in_channels=1, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.estimator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # [noise_level, blur_sigma, contrast_factor]
        )

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        params = self.estimator(feat)
        # 归一化到合理范围
        noise = torch.sigmoid(params[:, 0]) * 0.5  # [0, 0.5]
        blur = torch.sigmoid(params[:, 1]) * 5.0   # [0, 5]
        contrast = torch.sigmoid(params[:, 2]) * 2.0  # [0, 2]
        return torch.stack([noise, blur, contrast], dim=1)


class ConditionedResBlock(nn.Module):
    """
    条件残差块：根据退化参数调整特征
    使用FiLM (Feature-wise Linear Modulation) 思想
    """
    def __init__(self, channels, cond_dim=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # 条件投影：生成缩放和偏移
        self.cond_scale = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )
        self.cond_shift = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

        # 初始化最后一层为零，确保训练初期 FiLM 不改变特征
        nn.init.zeros_(self.cond_scale[-1].weight)
        nn.init.zeros_(self.cond_scale[-1].bias)
        nn.init.zeros_(self.cond_shift[-1].weight)
        nn.init.zeros_(self.cond_shift[-1].bias)

        # 实例归一化（对真实图像更稳定）
        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=False)

    def forward(self, x, cond):
        residual = x

        # 第一个卷积
        out = self.conv1(x)
        out = self.norm1(out)

        # FiLM调制
        scale = self.cond_scale(cond)[:, :, None, None]
        shift = self.cond_shift(cond)[:, :, None, None]
        out = out * (1 + scale) + shift

        out = F.relu(out)

        # 第二个卷积
        out = self.conv2(out)
        out = self.norm2(out)

        return out + residual


class SpatialAttention(nn.Module):
    """空间注意力：关注退化严重的区域"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.conv(x)
        return x * attn


class RealRestorer(nn.Module):
    """
    RealRestorer: Real-World Image Super-Resolution
    适配FY-4B真实卫星图像
    """
    def __init__(self, in_channels=1, num_features=64, num_blocks=4, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        # 退化估计
        self.deg_estimator = DegradationEstimator(in_channels)

        # 浅层特征提取
        self.shallow_conv = nn.Conv2d(in_channels, num_features, 3, padding=1)

        # 编码器残差块（条件化）
        self.enc_blocks = nn.ModuleList([
            ConditionedResBlock(num_features) for _ in range(num_blocks)
        ])

        # 空间注意力
        self.spatial_attn = SpatialAttention(num_features)

        # 中间过渡
        self.mid_conv = nn.Conv2d(num_features, num_features, 3, padding=1)

        # 解码器残差块（条件化）
        self.dec_blocks = nn.ModuleList([
            ConditionedResBlock(num_features) for _ in range(num_blocks)
        ])

        # 上采样
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        # 重建
        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if name == 'reconstruction.2':
                    # 最后一层输出小残差，初始化为零
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 估计退化参数
        deg_params = self.deg_estimator(x)

        # 浅层特征
        feat = F.relu(self.shallow_conv(x))

        # 编码器
        for block in self.enc_blocks:
            feat = block(feat, deg_params)

        # 空间注意力
        feat = self.spatial_attn(feat)

        # 中间过渡
        feat = self.mid_conv(feat)

        # 解码器
        for block in self.dec_blocks:
            feat = block(feat, deg_params)

        # 上采样
        feat = self.upsample(feat)

        # 重建
        sr = self.reconstruction(feat)

        # 全局残差
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                            mode='bicubic', align_corners=False)
        return sr + base


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    """训练一个epoch"""
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
    print("08_method_realrestorer - RealRestorer Real-World SR")
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
    print("\n创建 RealRestorer 模型...")
    model = RealRestorer(
        in_channels=1,
        num_features=64,
        num_blocks=4,
        upscale_factor=2
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)
    print(f"参数量: {model_params:,} ({model_size_mb:.2f} MB)")

    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 训练
    print(f"\n开始训练...")
    start_time = time.time()
    best_psnr = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device)
            best_psnr = max(best_psnr, metrics['psnr'])

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"train_loss: {train_loss:.4f} | "
                  f"val_psnr: {metrics['psnr']:.2f} dB | "
                  f"val_ssim: {metrics['ssim']:.4f} | "
                  f"lr: {current_lr:.6f}")

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
        "method": "08_method_realrestorer",
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
