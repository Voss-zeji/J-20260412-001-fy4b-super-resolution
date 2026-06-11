#!/usr/bin/env python3
"""
34_method_sfg_pftsr - Spatial-Frequency Gated PFTSR
基于 SFG-SwinIR 的频域门控思想 + PFTSR 骨干：
- 保留 PFTSR 空间分支（ResidualBlock + CBAM）
- 深层特征后插入 FrequencyGate 模块
- FFT 提取频域特征，可学习门控选择重要频率
"""

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='SFG-PFTSR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


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


class FrequencyGate(nn.Module):
    """频域门控模块（移植自 SFG-SwinIR）"""
    def __init__(self, channels):
        super().__init__()
        self.gate_gen = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Sigmoid()
        )
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        real = fft_feat.real
        imag = fft_feat.imag
        fft_cat = torch.cat([real, imag], dim=1)
        gate = self.gate_gen(fft_cat)
        freq_processed = self.freq_conv(fft_cat)
        freq_gated = freq_processed * gate
        freq_complex = torch.complex(freq_gated, torch.zeros_like(freq_gated))
        spatial = torch.fft.irfft2(freq_complex, s=(H, W), norm='ortho')
        return x + spatial


class SFGPFTSR(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feat=64, num_rb=4, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.shallow = nn.Sequential(
            nn.Conv2d(in_ch, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1)
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(feat) for _ in range(num_rb)])
        self.attn = ChannelAttention(feat)
        self.freq_gate = FrequencyGate(feat)
        self.upsample = nn.Sequential(
            nn.Conv2d(feat, feat * 4, 3, padding=1),
            nn.PixelShuffle(2)
        )
        self.reconstruct = nn.Conv2d(feat, out_ch, 3, padding=1)
        nn.init.constant_(self.reconstruct.weight, 0)

    def forward(self, x):
        feat = self.shallow(x)
        for rb in self.res_blocks:
            feat = rb(feat)
        feat = self.attn(feat)
        feat = self.freq_gate(feat)
        feat = self.upsample(feat)
        out = self.reconstruct(feat)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return out + base


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = torch.clamp(model(lr), -1, 1)
        loss = criterion(sr, hr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    psnr, ssim, rmse, n = 0, 0, 0, 0
    with torch.no_grad():
        for lr, hr, _ in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = torch.clamp(model(lr), -1, 1)
            for i in range(sr.size(0)):
                psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
                ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
                rmse += calculate_rmse(sr[i:i+1], hr[i:i+1])
            n += sr.size(0)
    return {'psnr': psnr/n, 'ssim': ssim/n, 'rmse': rmse/n}


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    channel = args.band.replace('CH', 'Channel')

    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir, high_res_dir=high_res_dir, channel=channel,
        batch_size=args.batch_size, num_workers=4, patch_size=64, upscale_factor=2
    )

    model = SFGPFTSR().to(device)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"[SFGPFTSR] Params: {model_params:,}")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_psnr, best_ssim, best_epoch = 0, 0, 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            best_ssim = metrics['ssim']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | "
                  f"Val PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f} | Best: {best_psnr:.2f}@{best_epoch}")

    if best_state:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, device)

    dummy = torch.randn(1, 1, 64, 64).to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad(): _ = model(dummy)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    infer_ms = (time.time() - t0) * 1000

    result = {
        "method": "34_method_sfg_pftsr",
        "band": args.band,
        "best_psnr": round(best_psnr, 4),
        "best_ssim": round(best_ssim, 4),
        "best_epoch": best_epoch,
        "final_psnr": round(final_metrics['psnr'], 4),
        "final_ssim": round(final_metrics['ssim'], 4),
        "train_epochs": args.epochs,
        "model_params": model_params,
        "inference_time_ms": round(infer_ms, 2),
        "train_time": round(time.time() - start_time, 2),
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone! Best PSNR: {best_psnr:.2f} dB @ epoch {best_epoch}")

if __name__ == '__main__':
    main()
