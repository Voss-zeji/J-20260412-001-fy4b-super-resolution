#!/usr/bin/env python3
"""
21_method_r2h_diff - R2H-Diff: RGB引导高光谱扩散模型

参考论文: arxiv 2605.05688
核心: 引导谱扩散模型用于高光谱重建

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
    parser = argparse.ArgumentParser(description='R2H-Diff')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.attn = SpectralAttention(out_ch)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return self.attn(x)


class R2HDiffNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        self.shallow = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        self.enc1 = UNetBlock(base_channels, base_channels)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.enc3 = UNetBlock(base_channels * 2, base_channels * 4)

        self.dec3 = UNetBlock(base_channels * 4, base_channels * 2)
        self.dec2 = UNetBlock(base_channels * 4, base_channels)
        self.dec1 = UNetBlock(base_channels * 2, base_channels)

        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.SiLU(inplace=True)
        )

        self.reconstruct = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.shallow(x)
        x1 = self.enc1(x1)

        x2 = F.avg_pool2d(x1, 2)
        x2 = self.enc2(x2)

        x3 = F.avg_pool2d(x2, 2)
        x3 = self.enc3(x3)

        d3 = self.dec3(x3)
        d3 = F.interpolate(d3, x2.shape[2:], mode='bilinear', align_corners=False)

        d2 = self.dec2(torch.cat([d3, x2], dim=1))
        d2 = F.interpolate(d2, x1.shape[2:], mode='bilinear', align_corners=False)

        d1 = self.dec1(torch.cat([d2, x1], dim=1))

        out = self.upsample(d1)
        out = self.reconstruct(out)

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

    model = R2HDiffNet().to(device)

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
        "method": "21_method_r2h_diff",
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
