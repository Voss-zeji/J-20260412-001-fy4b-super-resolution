# -*- coding: utf-8 -*-
"""
FY-4B 超分辨率实验主文件

这是 AI 进行修改的单一文件，包含:
- 模型定义 (PFT-SR)
- 训练循环
- 优化器配置
- 损失函数

使用方法:
    python main.py --band CH07          # 训练 CH07 通道
    python main.py --band CH08          # 训练 CH08 通道
    python main.py --band CH07 --test   # 测试模式 (不训练，只验证)

输出指标:
    val_psnr: 验证集 PSNR (越高越好)
    val_ssim: 验证集 SSIM
    train_time: 训练时间
"""

import os
import sys
import time
import argparse
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 导入固定工具 (不可修改)
from utils import (
    calculate_psnr, calculate_ssim, evaluate_model,
    save_checkpoint, load_checkpoint, cleanup_old_checkpoints,
    visualize_results, plot_training_curves
)
from data.fy4b_dataset import create_dataloaders


# ==================== 配置区域 (可修改) ====================

class Config:
    """实验配置 - 可在此修改超参数"""

    # 数据配置
    DATA_BASE_DIR = '/root/autodl-tmp/Calibration-FY4B'
    PATCH_SIZE = 64           # 低分辨率空间裁剪大小
    UPSCALE_FACTOR = 2        # 上采样因子 (固定2x)
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # 模型配置
    NUM_FEATURES = 64         # 特征维度
    NUM_PFT_BLOCKS = 3        # PFT块数量
    NUM_RB_PER_BLOCK = 3      # 每个PFT块的残差块数
    USE_ATTENTION = True      # 是否使用注意力

    # 损失函数权重
    LAMBDA_L1 = 1.0
    LAMBDA_SSIM = 0.5
    LAMBDA_FREQ = 0.1
    LAMBDA_GRAD = 0.1

    # 优化器配置
    LR = 0.0001
    WEIGHT_DECAY = 0.0001
    BETAS = (0.9, 0.999)

    # 训练配置
    NUM_EPOCHS = 100          # 训练轮数
    VAL_INTERVAL = 5          # 每N轮验证一次
    SAVE_INTERVAL = 10        # 每N轮保存一次
    GRAD_CLIP = 1.0           # 梯度裁剪

    # 早停配置
    EARLY_STOPPING = True
    PATIENCE = 10             # 容忍验证次数

    # 其他
    SEED = 42


# ==================== 模型定义 (可修改) ====================

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


# ==================== 损失函数 (可修改) ====================

class SSIMLoss(nn.Module):
    """SSIM损失"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self._create_window(window_size, 1)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class SRLoss(nn.Module):
    """超分辨率综合损失"""
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target):
        loss_dict = {}
        total_loss = 0.0

        if self.lambda_l1 > 0:
            l1 = self.l1_loss(pred, target)
            total_loss += self.lambda_l1 * l1
            loss_dict['l1'] = l1.item()

        if self.lambda_ssim > 0:
            ssim = self.ssim_loss(pred, target)
            total_loss += self.lambda_ssim * ssim
            loss_dict['ssim'] = ssim.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# ==================== 训练函数 (可修改) ====================

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_dict_sum = {}
    num_batches = len(train_loader)

    for batch_idx, (lr, hr, _) in enumerate(train_loader):
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()
        sr = model(lr)
        loss, loss_dict = criterion(sr, hr)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v

    avg_loss = total_loss / num_batches
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_sum.items()}
    return avg_loss, avg_loss_dict


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    loss_dict_sum = {}

    with torch.no_grad():
        for lr, hr, _ in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            loss, loss_dict = criterion(sr, hr)
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v

    avg_loss = total_loss / len(val_loader)
    avg_loss_dict = {k: v / len(val_loader) for k, v in loss_dict_sum.items()}
    metrics = evaluate_model(model, val_loader, device)

    return avg_loss, avg_loss_dict, metrics


class EarlyStopping:
    """早停模块"""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_psnr = 0.0
        self.early_stop = False

    def __call__(self, psnr):
        if psnr > self.best_psnr + self.min_delta:
            self.best_psnr = psnr
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True, False
            return False, False


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='FY-4B 超分辨率实验')
    parser.add_argument('--band', type=str, default='CH07', choices=['CH07', 'CH08'],
                        help='通道号 (CH07 或 CH08)')
    parser.add_argument('--test', action='store_true',
                        help='测试模式 (只验证，不训练)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 设置
    set_seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    band = args.band
    channel = band.replace('CH', 'Channel')
    low_res_dir = f"{Config.DATA_BASE_DIR}/4000M/{band}"
    high_res_dir = f"{Config.DATA_BASE_DIR}/2000M/{band}"
    output_dir = f"./results/{band}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"FY-4B 超分辨率实验 - {band}")
    print("=" * 60)
    print(f"任务: 4000M -> 2000M (4km -> 2km)")
    print(f"通道: {channel}")
    print(f"设备: {device}")
    print("=" * 60)

    # 创建数据加载器
    print("\n[1/4] 创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir,
        high_res_dir=high_res_dir,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        patch_size=Config.PATCH_SIZE,
        upscale_factor=Config.UPSCALE_FACTOR,
        channel=channel
    )
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")

    # 创建模型
    print("\n[2/4] 创建模型...")
    model = PFTSR(
        in_channels=1,
        out_channels=1,
        num_features=Config.NUM_FEATURES,
        num_pft_blocks=Config.NUM_PFT_BLOCKS,
        num_rb_per_block=Config.NUM_RB_PER_BLOCK,
        upscale_factor=Config.UPSCALE_FACTOR,
        use_attention=Config.USE_ATTENTION
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 测试模式
    if args.test:
        print("\n[3/4] 测试模式 - 验证当前模型...")
        criterion = SRLoss().to(device)
        val_loss, loss_dict, metrics = validate(model, val_loader, criterion, device)
        print(f"\n测试结果:")
        print(f"  val_psnr: {metrics['psnr']:.2f}")
        print(f"  val_ssim: {metrics['ssim']:.4f}")
        print(f"  val_loss: {val_loss:.6f}")
        return

    # 创建损失函数和优化器
    print("\n[3/4] 创建损失函数和优化器...")
    criterion = SRLoss(
        lambda_l1=Config.LAMBDA_L1,
        lambda_ssim=Config.LAMBDA_SSIM
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR,
        betas=Config.BETAS,
        weight_decay=Config.WEIGHT_DECAY
    )

    # 恢复训练
    start_epoch = 0
    best_psnr = 0.0
    if args.checkpoint:
        start_epoch, best_psnr = load_checkpoint(args.checkpoint, model, optimizer, device)
        start_epoch += 1
        print(f"  从 epoch {start_epoch} 恢复，当前最佳 PSNR: {best_psnr:.2f}")

    # 早停
    early_stopping = EarlyStopping(patience=Config.PATIENCE) if Config.EARLY_STOPPING else None

    # 训练循环
    print("\n[4/4] 开始训练...")
    print("=" * 60)
    print(f"Epochs: {start_epoch} -> {Config.NUM_EPOCHS}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LR}")
    print("=" * 60 + "\n")

    train_start_time = time.time()

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        epoch_start = time.time()

        # 训练
        train_loss, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.GRAD_CLIP
        )

        # 验证
        if (epoch + 1) % Config.VAL_INTERVAL == 0:
            val_loss, _, metrics = validate(model, val_loader, criterion, device)
            val_psnr = metrics['psnr']
            val_ssim = metrics['ssim']

            epoch_time = time.time() - epoch_start

            print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] "
                  f"train_loss: {train_loss:.4f} | "
                  f"val_psnr: {val_psnr:.2f} | "
                  f"val_ssim: {val_ssim:.4f} | "
                  f"time: {epoch_time:.1f}s")

            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                is_best = True
                print(f"  *** 新的最佳 PSNR: {best_psnr:.2f} dB ***")
            else:
                is_best = False

            # 保存检查点
            if (epoch + 1) % Config.SAVE_INTERVAL == 0 or is_best:
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                }
                save_checkpoint(state, output_dir, f'checkpoint_epoch_{epoch+1}.pth', is_best)

            # 早停检查
            if early_stopping:
                should_stop, _ = early_stopping(val_psnr)
                if should_stop:
                    print(f"\n早停触发! 连续 {Config.PATIENCE} 次验证无提升")
                    print(f"最佳 PSNR: {early_stopping.best_psnr:.2f} dB")
                    break

    # 训练结束
    train_time = time.time() - train_start_time

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳 val_psnr: {best_psnr:.2f} dB")
    print(f"总训练时间: {train_time:.1f}s")
    print(f"模型保存: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
