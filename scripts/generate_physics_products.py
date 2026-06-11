#!/usr/bin/env python3
"""
生成 PhysicsPFTSR 全圆盘产品 + 补全 EmambaIR summary
输出到 /root/autodl-tmp/products/ch07/
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/root/jobs/J-20260412-001-fy4b-super-resolution')
from utils import calculate_psnr, calculate_ssim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM_MIN, NORM_MAX = 150.0, 350.0

def normalize(img):
    img = (img - NORM_MIN) / (NORM_MAX - NORM_MIN)
    return img * 2 - 1

def denormalize(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    tensor = (tensor + 1) / 2.0
    tensor = tensor * (NORM_MAX - NORM_MIN) + NORM_MIN
    return tensor

# --- PFTSR model ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x): return x + self.conv2(self.relu(self.conv1(x)))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class PFTSR(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feat=64, num_rb=4, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.shallow = nn.Sequential(
            nn.Conv2d(in_ch, feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1))
        self.res_blocks = nn.ModuleList([ResidualBlock(feat) for _ in range(num_rb)])
        self.attn = ChannelAttention(feat)
        self.upsample = nn.Sequential(nn.Conv2d(feat, feat * 4, 3, padding=1), nn.PixelShuffle(2))
        self.reconstruct = nn.Conv2d(feat, out_ch, 3, padding=1)
        nn.init.constant_(self.reconstruct.weight, 0)
    def forward(self, x):
        feat = self.shallow(x)
        for rb in self.res_blocks: feat = rb(feat)
        feat = self.attn(feat); feat = self.upsample(feat); out = self.reconstruct(feat)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return out + base

def infer_full_disk(model, hdf_path):
    with h5py.File(hdf_path, 'r') as f:
        band_key = list(f.keys())[0]
        img = f[band_key][:].astype(np.float32)
    nan_mask = np.isnan(img)
    if nan_mask.any():
        valid_mean = np.nanmean(img)
        img = np.where(nan_mask, valid_mean, img)
    img_norm = normalize(img)
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    H, W = tensor.shape[2], tensor.shape[3]
    tile_size = 512; pad = 8
    sr = torch.zeros(1, 1, H * 2, W * 2, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H); x_end = min(x + tile_size, W)
                tile = tensor[:, :, y:y_end, x:x_end]
                tile_pad = F.pad(tile, (pad, pad, pad, pad), mode='reflect')
                tile_sr = model(tile_pad)
                tile_sr = tile_sr[:, :, pad:-pad, pad:-pad]
                sr[:, :, y*2:y_end*2, x*2:x_end*2] = tile_sr[:, :, :(y_end-y)*2, :(x_end-x)*2]
    sr = torch.clamp(sr, -1, 1)
    return sr.squeeze().cpu().numpy(), img

def generate_products_for_model(model, model_name, ckpt_path, product_dir, data_4km, data_2km, num_samples=10):
    print(f"\nGenerating products for {model_name}...")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()

    files_4km = sorted([f for f in os.listdir(data_4km) if f.endswith('.HDF')])
    files_2km = sorted([f for f in os.listdir(data_2km) if f.endswith('.HDF')])
    common = [f for f in files_4km if f in files_2km][:num_samples]

    subdir = product_dir / model_name
    subdir.mkdir(parents=True, exist_ok=True)

    results = []
    for fname in common:
        t0 = time.time()
        sr_norm, img_4km = infer_full_disk(model, data_4km / fname)
        img_4km_norm = normalize(img_4km)
        img_4km_t = torch.from_numpy(img_4km_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            bic_t = F.interpolate(img_4km_t, scale_factor=2, mode='bicubic', align_corners=False)
        bic_norm = bic_t.squeeze().cpu().numpy()
        highfreq_norm = sr_norm - bic_norm
        sr_phys = denormalize(sr_norm)
        bic_phys = denormalize(bic_norm)
        highfreq_phys = sr_phys - bic_phys

        with h5py.File(data_2km / fname, 'r') as f:
            band_key = list(f.keys())[0]
            hr = f[band_key][:].astype(np.float32)
            nan_mask = np.isnan(hr)
            if nan_mask.any(): hr = np.where(nan_mask, np.nanmean(hr), hr)
        h, w = hr.shape
        sr_crop = sr_phys[:h, :w]
        psnr = calculate_psnr(torch.from_numpy(normalize(sr_crop)).unsqueeze(0).unsqueeze(0),
                               torch.from_numpy(normalize(hr)).unsqueeze(0).unsqueeze(0))

        base = fname.replace('.HDF', '')
        np.save(subdir / f"{base}_SR.npy", sr_phys.astype(np.float32))
        np.save(subdir / f"{base}_bicubic.npy", bic_phys.astype(np.float32))
        np.save(subdir / f"{base}_highfreq.npy", highfreq_phys.astype(np.float32))

        results.append({"file": fname, "psnr": round(float(psnr), 2), "time_sec": round(time.time()-t0, 2)})
        print(f"  {fname}: PSNR={psnr:.2f}, time={time.time()-t0:.1f}s")

    with open(subdir / "product_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    avg = sum(r["psnr"] for r in results) / len(results)
    print(f"✅ {model_name} done, avg PSNR={avg:.2f}")

if __name__ == '__main__':
    CKPT_DIR = Path('/root/jobs/J-20260412-001-fy4b-super-resolution/checkpoints')
    PROD_DIR = Path('/root/autodl-tmp/products/ch07')
    DATA_4KM = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH07')
    DATA_2KM = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')

    # Generate PhysicsPFTSR products
    model = PFTSR(in_ch=1, out_ch=1, feat=64, num_rb=4, upscale=2)
    generate_products_for_model(model, "PhysicsPFTSR", CKPT_DIR / "PhysicsPFTSR_CH07_best.pth", PROD_DIR, DATA_4KM, DATA_2KM)

    print("\nAll products generated!")
