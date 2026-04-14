#!/usr/bin/env python3
"""
07_method_m2ir - M2IR: Mamba-style Modulation for Image Restoration

基于论文: "M2IR: Mamba-style Modulation for All-in-One Image Restoration", ArXiv 2603.14816
适配FY-4B: 使用Mamba状态空间模型进行长程建模

特点:
- Mamba状态空间模型（线性复杂度）
- 选择性扫描机制
- 适合处理大面积卫星图像
- 全局感受野
"""

import argparse
import json
import sys
import time
import math
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
    parser = argparse.ArgumentParser(description='M2IR Mamba SR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class SelectiveScan(nn.Module):
    """
    简化版选择性扫描（Mamba核心）
    参考Mamba论文的状态空间模型
    """
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state

        # 输入投影：生成x, B, C, delta
        self.in_proj = nn.Linear(dim, self.d_inner * 2 + d_state * 2 + self.d_inner, bias=False)

        # 卷积（局部建模）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=False
        )

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # 初始化参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        """
        x: [B, L, C]
        """
        B, L, C = x.shape

        # 输入投影
        xzBCd = self.in_proj(x)
        x_inner, z, B_ssm, C_ssm, delta = xzBCd.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.d_inner], dim=-1
        )

        # 卷积
        x_conv = self.conv1d(x_inner.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        # 简化版选择性扫描（使用softmax近似）
        A = -torch.exp(self.A_log.float())
        y = self.selective_scan(x_conv, delta, A, B_ssm, C_ssm, self.D)

        # 门控
        y = y * F.silu(z)

        # 输出投影
        output = self.out_proj(y)
        return output

    def selective_scan(self, x, delta, A, B, C, D):
        """简化的选择性扫描实现"""
        # 使用近似实现，实际部署可替换为mamba_ssm
        batch, seq_len, dim = x.shape

        # 离散化
        delta = F.softplus(delta)
        delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        delta_B = torch.einsum('bld,bln->bldn', delta, B)

        # 扫描
        h = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t, :].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            ys.append(y)
        y = torch.stack(ys, dim=1)

        return y + x * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    """Mamba块：核心构建单元"""
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = SelectiveScan(dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # 选择性扫描分支
        x = x + self.mixer(self.norm1(x))
        # MLP分支
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """图像patch嵌入"""
    def __init__(self, in_channels=1, patch_size=4, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H//patch, W//patch]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W//patch^2, embed_dim]
        return x


class PatchUnembed(nn.Module):
    """图像patch反嵌入"""
    def __init__(self, embed_dim=48, patch_size=4, out_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H // self.patch_size, W // self.patch_size)
        x = self.proj(x)
        return x


class M2IR(nn.Module):
    """
    M2IR: Mamba-style Modulation for Super-Resolution
    适配FY-4B卫星图像超分辨率
    """
    def __init__(self, in_channels=1, dim=48, num_blocks=6,
                 patch_size=4, upscale_factor=2, d_state=16):
        super().__init__()
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

        # Patch嵌入
        self.patch_embed = PatchEmbed(in_channels, patch_size, dim)

        # Mamba块序列
        self.blocks = nn.ModuleList([
            MambaBlock(dim, d_state=d_state) for _ in range(num_blocks)
        ])

        # 层归一化
        self.norm = nn.LayerNorm(dim)

        # Patch反嵌入后的处理
        self.upscale = nn.Sequential(
            nn.Conv2d(dim, dim * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        # 重建
        self.reconstruction = nn.Conv2d(dim, in_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if name == 'reconstruction':
                    # 最后一层输出小残差，初始化为零
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch嵌入
        x = self.patch_embed(x)  # [B, L, dim]

        # Mamba块处理
        for block in self.blocks:
            x = block(x)

        # 层归一化
        x = self.norm(x)

        # 反嵌入到空间
        x = x.transpose(1, 2).reshape(
            B, -1, H // self.patch_size, W // self.patch_size
        )

        # 上采样
        x = self.upscale(x)

        # 重建
        sr = self.reconstruction(x)

        # 全局残差
        base = F.interpolate(x[:, :C, :, :] if x.size(1) >= C else x[:, :1, :, :],
                            scale_factor=self.upscale_factor,
                            mode='bicubic', align_corners=False)
        if base.shape != sr.shape:
            base = F.interpolate(sr[:, :C, :, :], scale_factor=1, mode='nearest')

        return sr


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    for lr, hr, _ in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()
        sr = model(lr)

        # 确保尺寸匹配
        if sr.shape != hr.shape:
            sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)

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

            # 确保尺寸匹配
            if sr.shape != hr.shape:
                sr = F.interpolate(sr, size=hr.shape[2:], mode='bicubic', align_corners=False)

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
    print("07_method_m2ir - M2IR Mamba SR")
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
    print("\n创建 M2IR 模型...")
    model = M2IR(
        in_channels=1,
        dim=48,
        num_blocks=6,
        patch_size=4,
        upscale_factor=2,
        d_state=16
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)
    print(f"参数量: {model_params:,} ({model_size_mb:.2f} MB)")

    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
        "method": "07_method_m2ir",
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
