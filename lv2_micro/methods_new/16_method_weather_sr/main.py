#!/usr/bin/env python3
"""
16_method_weather_sr - 天气预报超分辨率

参考论文: arxiv 2409.11502
核心: 端到端深度学习方法学习 LR 到 HR 的气象图像映射

核心改动：
1. 网络结构：气象专用特征提取器，考虑气象数据的时间相关性
2. 气象感知模块：引入气象先验知识（云团、降水等特征）
3. 通道注意力：增强重要气象特征通道
4. 损失函数：结合 MSE 和气象一致性损失

数据形态适配：
- 输入: [B, 1, 64, 64] (低分辨率亮温图像)
- 输出: [B, 1, 128, 128] (超分辨率图像，x2)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='Weather Forecast SR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class WeatherAttention(nn.Module):
    """气象感知注意力模块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_attn = self.conv(x) * x
        sp_attn = self.spatial(ch_attn) * ch_attn
        return sp_attn


class WeatherFeatureExtractor(nn.Module):
    """气象特征提取器"""
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            WeatherAttention(out_channels),
        )

    def forward(self, x):
        return self.extractor(x)


class MeteorologyConsistencyLoss(nn.Module):
    """气象一致性损失"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # MSE基础损失
        mse = F.mse_loss(pred, target)

        # 局部对比度保持
        pred_local = self.local_contrast(pred)
        target_local = self.local_contrast(target)
        contrast_loss = F.mse_loss(pred_local, target_local)

        # 梯度一致性
        grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = F.mse_loss(grad_pred_x, grad_target_x) + F.mse_loss(grad_pred_y, grad_target_y)

        return mse + 0.2 * contrast_loss + 0.1 * grad_loss

    def local_contrast(self, x):
        """计算局部对比度"""
        pool = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        return torch.abs(x - pool)


class WeatherSRNet(nn.Module):
    """天气预报超分辨率网络"""
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        # 浅层特征提取
        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # 气象特征提取
        self.weather_feat = WeatherFeatureExtractor(base_channels, base_channels)

        # 深层特征提取（残差块）
        self.deep_feat = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(6)]
        )

        # 上采样
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # 重建
        self.reconstruct = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        shallow_feat = self.shallow(x)
        weather_feat = self.weather_feat(shallow_feat)
        deep_feat = self.deep_feat(weather_feat)

        # 残差连接
        feat = shallow_feat + deep_feat

        upsampled = self.upsample(feat)
        out = self.reconstruct(upsampled)

        # Bicubic残差
        bicubic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        out = out + bicubic

        return out


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)

        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            sr = model(lr)
            sr = torch.clamp(sr, -1, 1)

            for i in range(sr.size(0)):
                psnr = calculate_psnr(sr[i], hr[i], data_range=2.0)
                ssim = calculate_ssim(sr[i], hr[i], data_range=2.0)
                total_psnr += psnr
                total_ssim += ssim
                count += 1

    return total_psnr / count, total_ssim / count


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = WeatherSRNet(in_channels=1, out_channels=1, base_channels=64)
    model = model.to(device)

    # 加载数据
    train_loader, val_loader = create_dataloaders(
        band=args.band,
        patch_size=64,
        batch_size=args.batch_size,
        upscale_factor=2
    )

    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")

    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = MeteorologyConsistencyLoss()

    best_psnr = 0
    results = {
        "method": "16_method_weather_sr",
        "band": args.band,
        "epochs": 0,
        "best_psnr": 0,
        "best_ssim": 0,
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_psnr, val_ssim = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}] Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            results["best_psnr"] = best_psnr
            results["best_ssim"] = val_ssim
            results["best_epoch"] = epoch

        results["epochs"] = epoch

    results["status"] = "success"
    results["final_psnr"] = val_psnr
    results["final_ssim"] = val_ssim

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n训练完成! 最佳 PSNR: {best_psnr:.2f}, SSIM: {results['best_ssim']:.4f}")


if __name__ == "__main__":
    main()
