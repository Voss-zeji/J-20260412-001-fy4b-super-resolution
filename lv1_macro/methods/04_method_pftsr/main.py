#!/usr/bin/env python3
"""
04_method_pftsr - PFT-SR (Progressive Feature Transfer Super-Resolution)

基于渐进特征转移的自定义超分辨率网络
特点:
- 渐进式特征转移块 (PFT Block)
- 通道注意力 (CBAM) + 空间注意力
- 残差学习 + 全局残差连接
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
    parser = argparse.ArgumentParser(description='PFT-SR Method')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + identity


class ChannelAttention(nn.Module):
    """通道注意力"""
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
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
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


class ProgressiveFeatureTransferBlock(nn.Module):
    """渐进特征转移块 (PFT Block)"""
    def __init__(self, channels, upscale_factor=2, num_rb=3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # 残差块序列
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_rb)
        ])

        # 注意力模块
        if use_attention:
            self.attention = CBAM(channels)

        # 上采样层
        if upscale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            self.upsample = nn.Identity()

        # 特征融合
        self.fusion = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        feat = x

        # 通过残差块
        for rb in self.residual_blocks:
            feat = rb(feat)

        # 应用注意力
        if self.use_attention:
            feat = self.attention(feat)

        # 残差连接
        feat = feat + identity

        # 上采样
        upsampled = self.upsample(feat)
        upsampled = self.fusion(upsampled)

        return upsampled, feat


class PFTSR(nn.Module):
    """PFT-SR: Progressive Feature Transfer Network for Super-Resolution"""

    def __init__(self, in_channels=1, out_channels=1, num_features=64,
                 num_pft_blocks=3, num_rb_per_block=3, upscale_factor=2,
                 use_attention=True):
        super().__init__()

        self.upscale_factor = upscale_factor

        # 浅层特征提取
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )

        # PFT块序列
        self.pft_blocks = nn.ModuleList([
            ProgressiveFeatureTransferBlock(
                num_features,
                upscale_factor=upscale_factor if i == num_pft_blocks - 1 else 1,
                num_rb=num_rb_per_block,
                use_attention=use_attention
            ) for i in range(num_pft_blocks)
        ])

        # 全局残差学习
        self.global_residual = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 浅层特征提取
        shallow_feat = self.shallow_feat(x)

        # 渐进特征转移
        feat = shallow_feat
        for pft_block in self.pft_blocks:
            upsampled_feat, _ = pft_block(feat)
            feat = upsampled_feat

        # 全局残差学习
        sr_img = self.global_residual(feat)

        # 全局残差连接
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                            mode='bilinear', align_corners=False)
        return sr_img + base


class SRLoss(nn.Module):
    """超分辨率综合损失"""
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        loss = self.lambda_l1 * self.l1_loss(pred, target)
        # SSIM 损失在 utils 中实现，这里简化为L1
        return loss


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
    print("04_method_pftsr - PFT-SR")
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
    print("\n创建 PFT-SR 模型...")
    model = PFTSR(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_pft_blocks=3,
        num_rb_per_block=3,
        upscale_factor=2,
        use_attention=True
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
        "method": "04_method_pftsr",
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
