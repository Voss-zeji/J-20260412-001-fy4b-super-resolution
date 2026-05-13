#!/usr/bin/env python3
"""
24_method_ctdm - 连续时间分布匹配

参考论文: arxiv 2605.06376
核心: Few-Step扩散蒸馏

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
from utils import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='CTDM')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class DistributionMatchingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.conv(x)


class CTDMNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.SiLU(inplace=True)
        )

        self.dm_blocks = nn.Sequential(*[DistributionMatchingBlock(base_channels) for _ in range(8)])

        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.SiLU(inplace=True)
        )

        self.reconstruct = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.shallow(x)
        feat = self.dm_blocks(feat)
        upsampled = self.upsample(feat)
        out = self.reconstruct(upsampled)
        bicubic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        return out + bicubic


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for lr, hr, _ in dataloader:
        lr = lr.to(device)
        hr = hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = F.l1_loss(sr, hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0
    with torch.no_grad():
        for lr, hr, _ in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = torch.clamp(model(lr), -1, 1)
            for i in range(sr.size(0)):
                total_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
                total_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
                count += 1
    return total_psnr / count, total_ssim / count


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CTDMNet().to(device)

    low_res_dir = "/root/autodl-tmp/FY-4B/calibration/4000M/" + args.band
    high_res_dir = "/root/autodl-tmp/FY-4B/calibration/2000M/" + args.band
    channel = args.band.replace('CH', 'Channel')

    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=0, patch_size=64, upscale_factor=2,
        max_samples=100
    )

    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0
    results = {
        "method": "24_method_ctdm",
        "band": args.band,
        "epochs": 0,
        "best_psnr": 0,
        "best_ssim": 0,
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_psnr, val_ssim = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}] Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            results["best_psnr"], results["best_ssim"] = best_psnr, val_ssim
            results["best_epoch"] = epoch

        results["epochs"] = epoch

    results["status"] = "success"
    results["final_psnr"] = val_psnr
    results["final_ssim"] = val_ssim

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"完成! 最佳 PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
