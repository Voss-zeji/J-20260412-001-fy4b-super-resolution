#!/usr/bin/env python3
"""
15_method_ntire2026_ir_sr - NTIRE 2026 遥感红外超分辨率挑战赛方法

参考论文: arxiv 2604.21312
核心: 红外图像 x4 超分辨率，CNN/Transformer/混合架构

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
    parser = argparse.ArgumentParser(description='NTIRE2026 IR SR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.eca = ECA(channels)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)
        return out + residual


class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, 1, 2)
        self.conv7x7 = nn.Conv2d(channels, channels, 7, 1, 3)
        self.act = nn.PReLU(channels)
        self.fusion = nn.Conv2d(channels * 3, channels, 1, 1, 0)

    def forward(self, x):
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        f7 = self.conv7x7(x)
        out = torch.cat([f3, f5, f7], dim=1)
        out = self.act(self.fusion(out))
        return out


class NTIRE2026Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_res_blocks=8):
        super().__init__()

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.PReLU(base_channels)
        )

        self.msff = MultiScaleBlock(base_channels)

        self.res_blocks = nn.ModuleList([
            ResBlock(base_channels) for _ in range(num_res_blocks)
        ])

        self.global_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.PReLU(base_channels),
            ECA(base_channels)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(base_channels),
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(base_channels)
        )

        self.reconstruct = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        shallow_feat = self.shallow(x)
        ms_feat = self.msff(shallow_feat)
        res_feat = shallow_feat
        for block in self.res_blocks:
            res_feat = block(res_feat)
        fused = self.global_fusion(torch.cat([ms_feat, res_feat], dim=1))
        upsampled = self.upsample(fused)
        out = self.reconstruct(upsampled)
        bicubic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        return out + bicubic


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
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
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            sr = torch.clamp(model(lr), -1, 1)
            for i in range(sr.size(0)):
                total_psnr += calculate_psnr(sr[i], hr[i], data_range=2.0)
                total_ssim += calculate_ssim(sr[i], hr[i], data_range=2.0)
                count += 1
    return total_psnr / count, total_ssim / count


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = NTIRE2026Net(in_channels=1, out_channels=1, base_channels=64, num_res_blocks=8)
    model = model.to(device)

    channel = 'Channel07' if args.band == 'CH07' else 'Channel08'
    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=4, patch_size=64, upscale_factor=2
    )

    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0
    results = {
        "method": "15_method_ntire2026_ir_sr",
        "band": args.band,
        "epochs": 0,
        "best_psnr": 0,
        "best_ssim": 0,
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
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
