#!/usr/bin/env python3
"""
13_method_mambapft - MambaPFT

融合路径：M2IR (修复后的 2D Mamba SSM) + PFTSR (渐进特征转移 + CBAM)

核心改动（2处以上）：
1. 网络结构：重新设计 2D 选择性扫描（SelectiveScan2D），将原始 M2IR 失败的 1D
   序列化循环扫描改为双正交方向扫描（水平行扫描 + 垂直列扫描），保留 2D 空间
   邻域关系；两个方向的结果通过可学习 1x1 卷积融合。
2. 模型设计：不直接用 Mamba 替代 CNN，而是将 MambaBlock2D 作为"全局上下文增强
   瓶颈模块"插入到 PFTSR 的渐进特征转移块序列之后，形成"局部 CNN 提取细节 +
   全局 Mamba 建模长程依赖"的混合架构。
3. 模型设计：在 CNN 局部特征与 Mamba 全局特征之间引入门控融合（Gated Fusion），
   让网络自适应地决定每个空间位置应更多地依赖局部卷积特征还是全局状态空间特征。

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
    parser = argparse.ArgumentParser(description='MambaPFT')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class ResidualBlock(nn.Module):
    """标准残差块"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + identity


class ChannelAttention(nn.Module):
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
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class ProgressiveFeatureTransferBlock(nn.Module):
    """渐进特征转移块（来自 PFTSR）"""
    def __init__(self, channels, upscale_factor=2, num_rb=3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_rb)
        ])

        if use_attention:
            self.attention = CBAM(channels)

        if upscale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            self.upsample = nn.Identity()

        self.fusion = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        feat = x

        for rb in self.residual_blocks:
            feat = rb(feat)

        if self.use_attention:
            feat = self.attention(feat)

        feat = feat + identity
        upsampled = self.upsample(feat)
        upsampled = self.fusion(upsampled)
        return upsampled, feat


class SelectiveScan2D(nn.Module):
    """
    2D 选择性扫描：双正交方向（水平 + 垂直）
    修复原始 M2IR 1D 展平扫描丢失空间结构的问题。
    """
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state

        # 输入投影
        self.in_proj = nn.Linear(dim, self.d_inner * 2 + d_state * 2 + self.d_inner, bias=False)

        # 局部卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=False
        )

        # 方向融合
        self.dir_fusion = nn.Conv1d(self.d_inner * 2, self.d_inner, 1, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def _scan_1d(self, x, delta, A, B, C, D):
        """沿一个1D方向执行选择性扫描"""
        batch, seq_len, dim = x.shape
        delta = F.softplus(delta)
        delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        delta_B = torch.einsum('bld,bln->bldn', delta, B)

        h = torch.zeros(batch, dim, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t, :].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + x * D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        B, H, W, C = x.shape

        # 输入投影
        xzBCd = self.in_proj(x)
        x_inner, z, B_ssm, C_ssm, delta = xzBCd.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.d_inner], dim=-1
        )

        # 局部卷积（对每个空间位置沿通道做1D卷积）
        B_hw = B * H * W
        x_conv = self.conv1d(x_inner.reshape(B_hw, self.d_inner, 1)).reshape(B, H, W, self.d_inner)
        x_conv = F.silu(x_conv)

        A = -torch.exp(self.A_log.float())

        # 方向1: 水平扫描（行内，W方向）
        x_h = x_conv.reshape(B * H, W, self.d_inner)
        delta_h = delta.reshape(B * H, W, self.d_inner)
        B_h = B_ssm.reshape(B * H, W, self.d_state)
        C_h = C_ssm.reshape(B * H, W, self.d_state)
        y_h = self._scan_1d(x_h, delta_h, A, B_h, C_h, self.D)
        y_h = y_h.reshape(B, H, W, self.d_inner)

        # 方向2: 垂直扫描（列内，H方向）
        x_v = x_conv.permute(0, 2, 1, 3).reshape(B * W, H, self.d_inner)
        delta_v = delta.permute(0, 2, 1, 3).reshape(B * W, H, self.d_inner)
        B_v = B_ssm.permute(0, 2, 1, 3).reshape(B * W, H, self.d_state)
        C_v = C_ssm.permute(0, 2, 1, 3).reshape(B * W, H, self.d_state)
        y_v = self._scan_1d(x_v, delta_v, A, B_v, C_v, self.D)
        y_v = y_v.reshape(B, W, H, self.d_inner).permute(0, 2, 1, 3)

        # 方向融合
        y_cat = torch.cat([y_h, y_v], dim=-1)  # [B, H, W, 2*d_inner]
        y_fused = self.dir_fusion(y_cat.reshape(B_hw, self.d_inner * 2, 1)).reshape(B, H, W, self.d_inner)

        # 门控
        y_fused = y_fused * F.silu(z)

        # 输出投影
        output = self.out_proj(y_fused)
        return output


class MambaBlock2D(nn.Module):
    """2D Mamba 块"""
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = SelectiveScan2D(dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GatedFusion(nn.Module):
    """CNN 局部特征与 Mamba 全局特征的门控融合"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, local_feat, global_feat):
        concat = torch.cat([local_feat, global_feat], dim=1)
        g = self.gate(concat)
        return g * global_feat + (1 - g) * local_feat


class MambaPFT(nn.Module):
    """
    MambaPFT: PFTSR 局部渐进特征 + 2D Mamba 全局增强 + 门控融合
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64,
                 num_pft_blocks=3, num_rb_per_block=3, num_mamba_blocks=2,
                 upscale_factor=2, use_attention=True):
        super().__init__()
        self.upscale_factor = upscale_factor

        # 浅层特征提取
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )

        # PFT 块序列（局部 CNN 特征提取）
        self.pft_blocks = nn.ModuleList([
            ProgressiveFeatureTransferBlock(
                num_features,
                upscale_factor=1,  # 先在低分辨率空间处理
                num_rb=num_rb_per_block,
                use_attention=use_attention
            ) for _ in range(num_pft_blocks)
        ])

        # Mamba 全局增强瓶颈（2D 空间）
        self.mamba_blocks = nn.ModuleList([
            MambaBlock2D(num_features, d_state=16) for _ in range(num_mamba_blocks)
        ])
        self.mamba_norm = nn.LayerNorm(num_features)

        # 门控融合
        self.gated_fusion = GatedFusion(num_features)

        # 上采样层
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

        # 重建
        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                if name == 'reconstruction.2':
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 浅层特征
        shallow_feat = self.shallow_feat(x)

        # PFT 局部特征提取
        feat = shallow_feat
        for pft_block in self.pft_blocks:
            upsampled_feat, _ = pft_block(feat)
            feat = upsampled_feat

        # 保存局部特征
        local_feat = feat
        B, C, H, W = feat.shape

        # Mamba 全局增强（在 2D 空间上）
        mamba_in = feat.permute(0, 2, 3, 1)  # [B, H, W, C]
        for block in self.mamba_blocks:
            mamba_in = block(mamba_in)
        mamba_in = self.mamba_norm(mamba_in)
        global_feat = mamba_in.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 门控融合
        feat = self.gated_fusion(local_feat, global_feat)

        # 上采样与重建
        feat = self.upsample(feat)
        sr = self.reconstruction(feat)

        base = F.interpolate(x, scale_factor=self.upscale_factor,
                            mode='bicubic', align_corners=False)
        return sr + base


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
    print("13_method_mambapft - MambaPFT")
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

    model = MambaPFT(
        in_channels=1, out_channels=1, num_features=64,
        num_pft_blocks=3, num_rb_per_block=3, num_mamba_blocks=2,
        upscale_factor=2, use_attention=True
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
        "method": "13_method_mambapft",
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
