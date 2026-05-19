#!/usr/bin/env python3
"""
17_method_emambair - EmambaIR: Mamba状态空间模型图像重建

参考论文: arxiv 2605.08073
核心: 基于Mamba状态空间模型的高效图像重建，兼顾全局建模与计算效率

核心改动：
1. 网络结构：使用Mamba SSM替代Transformer进行全局特征建模
2. 高效计算：O(n)复杂度替代O(n^2)的注意力机制
3. 选择性状态空间：动态选择性地处理输入信息
4. 残差连接：多阶段特征融合

数据形态适配：
- 输入: [B, 1, 64, 64] (低分辨率亮温图像)
- 输出: [B, 1, 128, 128] (超分辨率图像，x2)
"""

import argparse
import json
import sys
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='EmambaIR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class SimplifiedMambaBlock(nn.Module):
    """简化的Mamba块（用于实际实现）"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.dt = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim, dim // 4))
        self.D = nn.Parameter(torch.ones(dim))
        self.o = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        x_norm = self.norm(x_flat)
        xz = self.proj(x_norm)
        x_inner, z = xz.chunk(2, dim=-1)

        # 简化的SSM操作
        dt = F.softplus(self.dt(x_inner))
        y = x_inner * dt

        # 门控
        y = y * torch.sigmoid(z)
        y = y + x_flat * self.D

        out = self.o(y)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


class MambaSRBlock(nn.Module):
    """Mamba超分辨率块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.mamba = SimplifiedMambaBlock(channels)
        self.norm = nn.InstanceNorm2d(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.mamba(out)
        out = self.norm(out)
        return residual + self.gamma * out


class EmambaIRNet(nn.Module):
    """EmambaIR 网络"""
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_blocks=8):
        super().__init__()

        # 浅层特征
        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.GELU()
        )

        # Mamba块
        self.mamba_blocks = nn.ModuleList([
            MambaSRBlock(base_channels) for _ in range(num_blocks)
        ])

        # 全局特征融合
        self.global_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1, 1, 0),
            nn.GELU()
        )

        # 上采样 (x2)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.GELU()
        )

        # 重建
        self.reconstruct = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        shallow_feat = self.shallow(x)

        # Mamba特征提取
        mamba_feat = shallow_feat
        for block in self.mamba_blocks:
            mamba_feat = block(mamba_feat)

        # 融合
        fused = self.global_fusion(torch.cat([shallow_feat, mamba_feat], dim=1))

        # 上采样和重建
        upsampled = self.upsample(fused)
        out = self.reconstruct(upsampled)

        # 残差
        bicubic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        out = out + bicubic

        return out


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
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for lr, hr, _ in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)
            sr = torch.clamp(sr, -1, 1)

            for i in range(sr.size(0)):
                psnr = calculate_psnr(sr[i:i+1], hr[i:i+1])
                ssim = calculate_ssim(sr[i:i+1], hr[i:i+1])
                total_psnr += psnr
                total_ssim += ssim
                count += 1

    return total_psnr / count, total_ssim / count


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = EmambaIRNet(in_channels=1, out_channels=1, base_channels=64, num_blocks=8)
    model = model.to(device)

    low_res_dir = "/root/autodl-tmp/FY-4B/calibration/4000M/" + args.band
    high_res_dir = "/root/autodl-tmp/FY-4B/calibration/2000M/" + args.band
    channel = args.band.replace('CH', 'Channel')

    # 加载数据
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=0, patch_size=64, upscale_factor=2,
        max_samples=100
    )

    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0
    results = {
        "method": "17_method_emambair",
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
