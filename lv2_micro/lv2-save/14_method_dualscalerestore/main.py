#!/usr/bin/env python3
"""
14_method_dualscalerestore - DualScaleRestorer

融合路径：EDSR (修复后的深层残差分支) + TinyNina (轻量快速分支) + RealRestorer (退化感知融合)

核心改动（2处以上）：
1. 网络结构：设计双尺度并行分支——DeepBranch（EDSR风格深层残差，64通道×16块）
   与 LightBranch（TinyNina风格轻量卷积，32通道×2块）同时处理输入，分别负责
   长程细节恢复和边缘快速响应。
2. 模型设计：引入退化感知的动态融合门（Dynamic Fusion Gate），使用 RealRestorer
   的 DegradationEstimator 预测两个分支的融合权重 α，根据输入退化程度自适应
   调整对深层分支和轻量分支的依赖比例。
3. 模型设计：增加中期跨分支 Cross-Attention 交互模块，在上采样前的瓶颈层将
   LightBranch 的锐化边缘信息注入 DeepBranch，同时 DeepBranch 的上下文信息
   反馈到 LightBranch，实现特征互补。

数据形态适配：
- 输入: [B, 1, 64, 64]
- 输出: [B, 1, 128, 128]
- 使用 create_dataloaders(..., patch_size=64, upscale_factor=2)
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='DualScaleRestorer')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class EDSRResidualBlock(nn.Module):
    """EDSR 残差块（无 BN，带残差缩放）"""
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        res = res * self.res_scale
        return x + res


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class LightResidualBlock(nn.Module):
    """TinyNina 轻量残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.gate = ChannelGate(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.gate(out)
        return out + identity


class DegradationEstimator(nn.Module):
    """退化估计器"""
    def __init__(self, in_channels=1, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.estimator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        params = self.estimator(feat)
        noise = torch.sigmoid(params[:, 0]) * 0.5
        blur = torch.sigmoid(params[:, 1]) * 5.0
        contrast = torch.sigmoid(params[:, 2]) * 2.0
        return torch.stack([noise, blur, contrast], dim=1)


class DynamicFusionGate(nn.Module):
    """根据退化参数预测双分支融合权重"""
    def __init__(self, cond_dim=3):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(cond_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )

    def forward(self, cond):
        logits = self.predictor(cond)
        alpha = F.softmax(logits, dim=-1)  # [B, 2]
        return alpha


class CrossAttentionFusion(nn.Module):
    """轻量 Cross-Attention：让分支B的信息注入分支A"""
    def __init__(self, dim_a, dim_b, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Conv2d(dim_a, dim_a, 1, bias=False)
        self.kv_proj = nn.Conv2d(dim_b, dim_a * 2, 1, bias=False)
        self.out_proj = nn.Conv2d(dim_a, dim_a, 1, bias=False)
        self.scale = (dim_a // num_heads) ** -0.5

    def forward(self, feat_a, feat_b):
        """
        feat_a: [B, dim_a, H, W]
        feat_b: [B, dim_b, H, W]
        """
        B, C, H, W = feat_a.shape
        q = self.q_proj(feat_a)  # [B, dim_a, H, W]
        kv = self.kv_proj(feat_b)  # [B, dim_a*2, H, W]
        k, v = kv.chunk(2, dim=1)

        # reshape to [B, num_heads, HW, head_dim]
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(out)
        return feat_a + out


class DualScaleRestorer(nn.Module):
    """
    DualScaleRestorer: 双尺度并行分支 + 退化感知动态融合 + Cross-Attention 交互
    """
    def __init__(self, in_channels=1, out_channels=1, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        # 退化估计器
        self.degradation_estimator = DegradationEstimator(in_channels)

        # ================== DeepBranch (EDSR 风格) ==================
        self.deep_head = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.deep_body = nn.ModuleList([
            EDSRResidualBlock(64, res_scale=0.1) for _ in range(16)
        ])
        self.deep_body_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.deep_upsample = nn.Sequential(
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.deep_tail = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        # ================== LightBranch (TinyNina 风格) ==================
        self.light_shallow = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32),
            nn.ReLU(inplace=True),
            ChannelGate(32)
        )
        self.light_body = nn.ModuleList([
            LightResidualBlock(32) for _ in range(2)
        ])
        self.light_mid_conv = DepthwiseSeparableConv(32, 32)
        self.light_upsample = nn.Sequential(
            nn.Conv2d(32, 32 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.light_tail = DepthwiseSeparableConv(32, out_channels)

        # ================== 跨分支交互 ==================
        # LightBranch -> DeepBranch 的 Cross-Attention
        self.cross_attn_light2deep = CrossAttentionFusion(dim_a=64, dim_b=32, num_heads=4)
        # DeepBranch -> LightBranch 的通道注意力引导
        self.deep_guide_light = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.Sigmoid()
        )

        # ================== 动态融合 ==================
        self.fusion_gate = DynamicFusionGate(cond_dim=3)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if name in ['deep_tail', 'light_tail.pointwise']:
                    # 极小值初始化而非严格零
                    nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 退化参数估计
        deg_params = self.degradation_estimator(x)

        # ================== DeepBranch 前向 ==================
        deep_feat = self.deep_head(x)
        deep_res = deep_feat
        for block in self.deep_body:
            deep_feat = block(deep_feat)
        deep_feat = self.deep_body_conv(deep_feat)
        deep_feat = deep_feat + deep_res

        # ================== LightBranch 前向（到中期） ==================
        light_feat = self.light_shallow(x)
        light_identity = light_feat
        for block in self.light_body:
            light_feat = block(light_feat)
        light_feat = self.light_mid_conv(light_feat) + light_identity

        # ================== 中期 Cross-Attention 交互 ==================
        # LightBranch 引导 DeepBranch
        deep_feat = self.cross_attn_light2deep(deep_feat, light_feat)
        # DeepBranch 门控引导 LightBranch
        light_guide = self.deep_guide_light(deep_feat)
        light_feat = light_feat * light_guide

        # ================== 各自上采样 ==================
        deep_sr = self.deep_upsample(deep_feat)
        deep_sr = self.deep_tail(deep_sr)

        light_sr = self.light_upsample(light_feat)
        light_sr = self.light_tail(light_sr)

        # ================== 退化感知动态融合 ==================
        alpha = self.fusion_gate(deg_params)  # [B, 2]
        alpha_deep = alpha[:, 0:1, None, None]
        alpha_light = alpha[:, 1:2, None, None]

        fused_sr = alpha_deep * deep_sr + alpha_light * light_sr

        # 全局残差
        base = F.interpolate(x, scale_factor=self.upscale_factor,
                            mode='bicubic', align_corners=False)
        return fused_sr + base


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=1.0):
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
    model.eval()
    total_psnr = total_ssim = total_rmse = 0.0
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
    print("14_method_dualscalerestore - DualScaleRestorer")
    print("=" * 60)
    print(f"Band: {args.band}, Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    channel = args.band.replace('CH', 'Channel')

    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=4, patch_size=64, upscale_factor=2
    )
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    model = DualScaleRestorer(
        in_channels=1, out_channels=1, upscale_factor=2
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)
    print(f"参数量: {model_params:,} ({model_size_mb:.2f} MB)")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    best_psnr = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device)
            best_psnr = max(best_psnr, metrics['psnr'])
            print(f"Epoch [{epoch+1}/{args.epochs}] loss: {train_loss:.4f} | "
                  f"val_psnr: {metrics['psnr']:.2f} dB | val_ssim: {metrics['ssim']:.4f}")

    train_time = time.time() - start_time
    metrics = evaluate(model, val_loader, device)

    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time_ms = (time.time() - t0) * 1000

    result = {
        "method": "14_method_dualscalerestore",
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
    print(f"Best val_psnr: {best_psnr:.2f} dB")
    print(f"Final val_psnr: {result['val_psnr']:.2f} dB")
    print(f"Train time: {result['train_time']:.1f}s")
    print(f"结果已保存: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
