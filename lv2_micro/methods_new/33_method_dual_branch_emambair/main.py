#!/usr/bin/env python3
"""
33_method_dual_branch_emambair - Dual-Branch EmambaIR
基于 EmambaIR (当前 Top-1) + Dual-Branch 思想：
- 分支A：EmambaIR 空间分支（Mamba SSM）
- 分支B：频域分支（FFT + 轻量 CNN）
- 融合：通道注意力加权
"""

import argparse, json, sys, time, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='Dual-Branch EmambaIR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class SimplifiedMambaBlock(nn.Module):
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
        dt = F.softplus(self.dt(x_inner))
        y = x_inner * dt
        y = y * torch.sigmoid(z)
        y = y + x_flat * self.D
        out = self.o(y)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class MambaSRBlock(nn.Module):
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


class SpatialBranch(nn.Module):
    """空间分支：基于 EmambaIR"""
    def __init__(self, in_ch=1, feat=64, num_blocks=8):
        super().__init__()
        self.shallow = nn.Sequential(nn.Conv2d(in_ch, feat, 3, 1, 1), nn.GELU())
        self.mamba_blocks = nn.ModuleList([MambaSRBlock(feat) for _ in range(num_blocks)])
        self.global_fusion = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1, 1, 0), nn.GELU()
        )
    def forward(self, x):
        shallow = self.shallow(x)
        feat = shallow
        for block in self.mamba_blocks:
            feat = block(feat)
        fused = self.global_fusion(torch.cat([shallow, feat], dim=1))
        return fused


class FrequencyBranch(nn.Module):
    """频域分支：FFT + 轻量 CNN"""
    def __init__(self, in_ch=1, feat=32):
        super().__init__()
        self.shallow = nn.Conv2d(in_ch, feat, 3, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1),
        )
        self.out = nn.Conv2d(feat, feat, 3, padding=1)
    def forward(self, x):
        feat = self.shallow(x)
        fft_feat = torch.fft.rfft2(feat, norm='ortho')
        real, imag = fft_feat.real, fft_feat.imag
        fft_cat = torch.cat([real, imag], dim=1)
        fft_processed = self.conv1(fft_cat)
        fft_complex = torch.complex(fft_processed, torch.zeros_like(fft_processed))
        spatial = torch.fft.irfft2(fft_complex, s=feat.shape[2:], norm='ortho')
        spatial = self.conv2(spatial)
        return self.out(spatial)


class FusionModule(nn.Module):
    def __init__(self, feat_a=64, feat_b=32, out_feat=64):
        super().__init__()
        self.compress = nn.Conv2d(feat_a + feat_b, out_feat, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_feat, out_feat // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feat // 8, out_feat, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_feat, out_feat, 3, padding=1),
        )
    def forward(self, feat_a, feat_b):
        merged = torch.cat([feat_a, feat_b], dim=1)
        merged = self.compress(merged)
        merged = merged * self.attn(merged)
        return self.refine(merged)


class DualBranchEmambaIR(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.spatial_branch = SpatialBranch(in_ch, feat=64, num_blocks=8)
        self.freq_branch = FrequencyBranch(in_ch, feat=32)
        self.fusion = FusionModule(64, 32, 64)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU()
        )
        self.reconstruct = nn.Conv2d(64, out_ch, 3, padding=1)
        nn.init.constant_(self.reconstruct.weight, 0)
    def forward(self, x):
        feat_s = self.spatial_branch(x)
        feat_f = self.freq_branch(x)
        feat = self.fusion(feat_s, feat_f)
        feat = self.upsample(feat)
        out = self.reconstruct(feat)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
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

    model = DualBranchEmambaIR().to(device)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"[DualBranchEmambaIR] Params: {model_params:,}")

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
        "method": "33_method_dual_branch_emambair",
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
    torch.save(model.state_dict(), args.output.replace(".json", "_best.pth"))
    print(f"\nDone! Best PSNR: {best_psnr:.2f} dB @ epoch {best_epoch}")

if __name__ == '__main__':
    main()
