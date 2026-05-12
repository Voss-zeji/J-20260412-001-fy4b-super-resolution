#!/usr/bin/env python3
"""
28_method_crossview_3d - 跨视图3D重建

参考论文: arxiv 2605.07978
核心: 卫星-无人机-地面多视角重建

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
    parser = argparse.ArgumentParser(description='CrossView 3D')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class CrossViewBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(self.conv(x)) + x


class CrossView3DNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.crossview_blocks = nn.Sequential(*[CrossViewBlock(base_channels) for _ in range(6)])

        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.reconstruct = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.shallow(x)
        feat = self.crossview_blocks(feat)
        upsampled = self.upsample(feat)
        out = self.reconstruct(upsampled)
        bicubic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        return out + bicubic


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        lr, hr = batch['lr'].to(device), batch['hr'].to(device)
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
        for batch in dataloader:
            lr, hr = batch['lr'].to(device), batch['hr'].to(device)
            sr = torch.clamp(model(lr), -1, 1)
            for i in range(sr.size(0)):
                total_psnr += calculate_psnr(sr[i], hr[i], data_range=2.0)
                total_ssim += calculate_ssim(sr[i], hr[i], data_range=2.0)
                count += 1
    return total_psnr / count, total_ssim / count


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CrossView3DNet().to(device)
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=4, patch_size=64, upscale_factor=2
    ), 64, args.batch_size, 2)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr, results = 0, {"method": "28_method_crossview_3d", "band": ('Channel07' if args.band == 'CH07' else 'Channel08')}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_psnr, val_ssim = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}] Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            results["best_psnr"], results["best_ssim"] = best_psnr, val_ssim
            results["best_epoch"] = epoch

    results["status"] = "success"
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"完成! 最佳 PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
