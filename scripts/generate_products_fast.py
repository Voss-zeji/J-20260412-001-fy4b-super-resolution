#!/usr/bin/env python3
"""Full-disk product generation with patch-based inference to avoid OOM"""

import sys, json, os, time, importlib
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import h5py

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
PRODUCT_DIR = Path('/root/autodl-tmp/products/ch07')
DATA_4KM = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH07')
DATA_2KM = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM_MIN, NORM_MAX = 150.0, 350.0

PATCH_SIZE = 64  # LR patch size (matches training)
UPSCALE = 2
OVERLAP = 8      # overlap between patches (in LR space)


def denormalize(tensor):
    return (tensor + 1) / 2 * (NORM_MAX - NORM_MIN) + NORM_MIN


def load_hdf_data(filepath):
    with h5py.File(filepath, 'r') as f:
        for key in ['CH07', 'Channel07']:
            if key in f:
                data = f[key][()]
                break
    if np.any(~np.isfinite(data)):
        data = np.where(~np.isfinite(data), np.nanmean(data), data)
    return data.astype(np.float32)


def build_model(name):
    configs = {
        'EmambaIR': {
            'path': 'lv2_micro/lv2-save/17_method_emambair',
            'class': 'EmambaIRNet',
            'kwargs': {'in_channels': 1, 'out_channels': 1, 'base_channels': 64, 'num_blocks': 8},
            'ckpt': CHECKPOINT_DIR / 'EmambaIR_CH07_best.pth',
        },
        'SFGSwinIR': {
            'path': 'lv2_micro/methods_new/31_method_sfg_swinir',
            'class': 'SFGSwinIR',
            'kwargs': {'in_channels': 1, 'out_channels': 1, 'embed_dim': 60,
                       'depths': [4, 4], 'num_heads': [4, 4],
                       'window_size': 8, 'mlp_ratio': 2.0, 'upscale_factor': 2},
            'ckpt': CHECKPOINT_DIR / 'SFGSwinIR_CH07_best.pth',
        },
        'PhysicsPFTSR': {
            'path': 'lv2_micro/methods_new/32_method_physics_pftsr',
            'class': 'PFTSR',
            'kwargs': {},
            'ckpt': CHECKPOINT_DIR / 'PhysicsPFTSR_CH07_best.pth',
        },
    }
    cfg = configs[name]
    mod_path = PROJECT_ROOT / cfg['path'] / 'main.py'
    spec = importlib.util.spec_from_file_location(f'mod_{name}', mod_path)
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(PROJECT_ROOT / cfg['path']))
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        spec.loader.exec_module(mod)
        Cls = getattr(mod, cfg['class'])
        model = Cls(**cfg['kwargs'])
        state = torch.load(cfg['ckpt'], map_location=DEVICE)
        model.load_state_dict(state)
        model = model.to(DEVICE)
        nparams = sum(p.numel() for p in model.parameters())
        print(f'  [OK] {name}: {nparams:,} params')
        return model
    finally:
        sys.path = old_path


@torch.no_grad()
def infer_patch(model, lr_patch):
    """Infer on a single LR patch, return SR patch as numpy"""
    t = torch.from_numpy(lr_patch[np.newaxis, np.newaxis, ...]).float().to(DEVICE)
    t = (t - NORM_MIN) / (NORM_MAX - NORM_MIN) * 2 - 1
    sr = model(t)
    sr_phys = denormalize(sr)
    return sr_phys.squeeze().cpu().numpy()


def patch_inference_full_disk(model, lr_full):
    """Split full-disk LR image into overlapping patches, infer each, stitch back"""
    H, W = lr_full.shape
    pH, pW = PATCH_SIZE, PATCH_SIZE
    stride = pH - OVERLAP

    # Pad LR to multiples of stride
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    lr_padded = np.pad(lr_full, ((0, pad_h), (0, pad_w)), mode='reflect')
    Hp, Wp = lr_padded.shape

    # Output size in SR space (with padding)
    out_h = (Hp - pH) // stride * UPSCALE * pH // (pH - OVERLAP) + UPSCALE * pH
    # Actually easier: compute output dimensions directly
    n_h = (Hp - pH) // stride + 1
    n_w = (Wp - pW) // stride + 1

    sr_full = np.zeros((Hp * UPSCALE, Wp * UPSCALE), dtype=np.float32)
    weight_map = np.zeros((Hp * UPSCALE, Wp * UPSCALE), dtype=np.float32)

    # Robust triangular blending weight per patch
    spp = PATCH_SIZE * UPSCALE  # 128
    wy = 1 - np.abs(np.linspace(-1, 1, spp))  # triangle: 0 at edges, 1 at center
    wx = 1 - np.abs(np.linspace(-1, 1, spp))
    patch_weight = np.outer(wy, wx)

    for i in range(n_h):
        for j in range(n_w):
            y = i * stride
            x = j * stride
            lr_patch = lr_padded[y:y+pH, x:x+pW]
            sr_patch = infer_patch(model, lr_patch)

            sy, sx = y * UPSCALE, x * UPSCALE
            sr_full[sy:sy+spp, sx:sx+spp] += sr_patch * patch_weight
            weight_map[sy:sy+spp, sx:sx+spp] += patch_weight

    weight_map = np.maximum(weight_map, 1e-10)
    sr_full /= weight_map

    # Crop back to original size
    sr_full = sr_full[:H * UPSCALE, :W * UPSCALE]
    return sr_full


@torch.no_grad()
def generate_products(model, name, num_samples=10):
    print(f'\n=== Generating products: {name} ===')
    files_4km = sorted([f for f in os.listdir(DATA_4KM) if f.endswith('.HDF')])
    files_2km = sorted([f for f in os.listdir(DATA_2KM) if f.endswith('.HDF')])
    common = [f for f in files_4km if f in files_2km][:num_samples]
    print(f'  Processing {len(common)} samples (patch size={PATCH_SIZE}, overlap={OVERLAP})')

    out_dir = PRODUCT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    results = []
    for fname in common:
        lr = load_hdf_data(str(DATA_4KM / fname))
        hr = load_hdf_data(str(DATA_2KM / fname))

        t0 = time.time()
        sr = patch_inference_full_disk(model, lr)
        dt = time.time() - t0

        # PSNR in physical space
        hr_t = torch.from_numpy(hr[np.newaxis, np.newaxis, ...]).float()
        sr_t = torch.from_numpy(sr[np.newaxis, np.newaxis, ...]).float()
        mse = F.mse_loss(sr_t, hr_t)
        psnr = 20 * torch.log10(300.0 / torch.sqrt(mse)).item()

        np.save(out_dir / f'{Path(fname).stem}_SR.npy', sr.astype(np.float32))
        results.append({'file': fname, 'psnr': round(psnr, 2), 'time_s': round(dt, 2)})
        print(f'  {fname}: PSNR={psnr:.2f} dB, time={dt:.1f}s')

        # Clear GPU cache between samples
        torch.cuda.empty_cache()

    avg = sum(r['psnr'] for r in results) / len(results)
    print(f'  Done {name}, avg PSNR={avg:.2f} dB')
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({'model': name, 'avg_psnr': round(avg, 2), 'results': results}, f, indent=2)
    return results


def main():
    print('=' * 60)
    print('CH07 Product Generation (patch-based full-disk inference)')
    print(f'Device: {DEVICE}')
    print(f'Products -> {PRODUCT_DIR}')
    print(f'Patch: {PATCH_SIZE}x{PATCH_SIZE}, Overlap: {OVERLAP}')
    print('=' * 60)

    for name in ['EmambaIR', 'SFGSwinIR', 'PhysicsPFTSR']:
        model = build_model(name)
        if model:
            generate_products(model, name, num_samples=10)
            del model
            torch.cuda.empty_cache()

    print('\n' + '=' * 60)
    print('All done!')
    print(f'Products: {PRODUCT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
