#!/root/miniconda3/envs/mamba2/bin/python -u
"""Phase C3/C4: CH08 4km -> ~2km SR inference using adapted model"""

import sys, os, json, time, importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h5py

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
PRODUCT_DIR = Path('/root/autodl-tmp/products/ch08')
CH08_4K = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH08')
CH08_2K_FAKE = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH08')

sys.path.insert(0, str(PROJECT_ROOT))
from utils import calculate_psnr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM_MIN, NORM_MAX = 150.0, 350.0

def load_hdf(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['CH08'][()]
    return np.where(np.isfinite(data), data, np.nanmean(data)).astype(np.float32)

def load_adapted_model():
    mod_path = PROJECT_ROOT / 'lv2_micro' / 'lv2-save' / '17_method_emambair'
    spec = importlib.util.spec_from_file_location('emod', mod_path / 'main.py')
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(mod_path))
    sys.path.insert(0, str(PROJECT_ROOT))
    spec.loader.exec_module(mod)
    model = mod.EmambaIRNet(in_channels=1, out_channels=1, base_channels=64, num_blocks=8).to(DEVICE)
    ckpt = CHECKPOINT_DIR / 'CH08_adapt_best.pth'
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        print(f'Loaded adapted checkpoint: {ckpt}')
    else:
        state = torch.load(CHECKPOINT_DIR / 'EmambaIR_CH07_best.pth', map_location=DEVICE)
        model.load_state_dict(state)
        print('No adapted checkpoint, using CH07 base model')
    model.eval()
    sys.path = old_path
    return model

@torch.no_grad()
def infer_patch(model, lr_patch):
    t = torch.from_numpy(lr_patch[np.newaxis, np.newaxis, ...]).float().to(DEVICE)
    t = (t - NORM_MIN) / (NORM_MAX - NORM_MIN) * 2 - 1
    sr = model(t)
    sr = torch.clamp(sr, -1, 1)
    sr_phys = (sr + 1) / 2 * (NORM_MAX - NORM_MIN) + NORM_MIN
    return sr_phys.squeeze().cpu().numpy()

def patch_inference(model, lr_full):
    H, W = lr_full.shape
    ps = 64
    stride = 56
    spp = ps * 2
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    lr_pad = np.pad(lr_full, ((0, pad_h), (0, pad_w)), mode='reflect')
    Hp, Wp = lr_pad.shape
    n_h = (Hp - ps) // stride + 1
    n_w = (Wp - ps) // stride + 1
    sr_full = np.zeros((Hp * 2, Wp * 2), dtype=np.float32)
    wsum = np.zeros((Hp * 2, Wp * 2), dtype=np.float32)
    wy = 1 - np.abs(np.linspace(-1, 1, spp))
    wx = 1 - np.abs(np.linspace(-1, 1, spp))
    pw = np.outer(wy, wx)
    for i in range(n_h):
        for j in range(n_w):
            patch = lr_pad[i*stride:i*stride+ps, j*stride:j*stride+ps]
            sr_p = infer_patch(model, patch)
            sy, sx = i*stride*2, j*stride*2
            sr_full[sy:sy+spp, sx:sx+spp] += sr_p * pw
            wsum[sy:sy+spp, sx:sx+spp] += pw
    wsum = np.maximum(wsum, 1e-10)
    sr_full /= wsum
    return sr_full[:H*2, :W*2]

def main():
    print('=' * 60)
    print('Phase C3/C4: CH08 4km -> ~2km SR')
    print(f'Device: {DEVICE}')
    print(f'Products -> {PRODUCT_DIR}')
    print('=' * 60)

    model = load_adapted_model()
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in os.listdir(CH08_4K) if f.endswith('.HDF')])
    n_samples = min(10, len(files))
    print(f'Generating {n_samples} CH08 SR products...')

    results = []
    for fname in files[:n_samples]:
        t0 = time.time()
        lr = load_hdf(str(CH08_4K / fname))
        sr = patch_inference(model, lr)
        dt = time.time() - t0

        # Bicubic baseline
        lr_t = torch.from_numpy(lr[np.newaxis, np.newaxis, ...]).float()
        bicubic_t = F.interpolate(lr_t, scale_factor=2, mode='bicubic', align_corners=False)
        bicubic_phys = bicubic_t.squeeze().numpy()

        # Proxy PSNR vs interpolated 2K (informational only, not ground truth)
        fake_2k = load_hdf(str(CH08_2K_FAKE / fname))
        sr_t = torch.from_numpy(sr[np.newaxis, np.newaxis, ...]).float()
        fake_t = torch.from_numpy(fake_2k[np.newaxis, np.newaxis, ...]).float()
        bicubic_t = torch.from_numpy(bicubic_phys[np.newaxis, np.newaxis, ...]).float()
        mse_sr = F.mse_loss(sr_t, fake_t)
        mse_bic = F.mse_loss(bicubic_t, fake_t)
        psnr_sr = 20 * torch.log10(300.0 / torch.sqrt(mse_sr)).item()
        psnr_bic = 20 * torch.log10(300.0 / torch.sqrt(mse_bic)).item()

        np.save(PRODUCT_DIR / f'{Path(fname).stem}_SR.npy', sr.astype(np.float32))
        results.append({
            'file': fname,
            'proxy_psnr_sr': round(psnr_sr, 2),
            'proxy_psnr_bicubic': round(psnr_bic, 2),
            'time_s': round(dt, 1)
        })
        print(f'  {fname}: SR PSNR={psnr_sr:.2f}, Bicubic={psnr_bic:.2f}, time={dt:.1f}s')

    avg_sr = np.mean([r['proxy_psnr_sr'] for r in results])
    avg_bic = np.mean([r['proxy_psnr_bicubic'] for r in results])
    print(f'\nAvg proxy PSNR: SR={avg_sr:.2f}, Bicubic={avg_bic:.2f}')
    print(f'Products saved: {PRODUCT_DIR}')

    with open(PRODUCT_DIR / 'results.json', 'w') as f:
        json.dump({
            'method': 'EmambaIR_CH08_adapted',
            'samples': n_samples,
            'avg_proxy_psnr_sr': round(avg_sr, 2),
            'avg_proxy_psnr_bicubic': round(avg_bic, 2),
            'results': results,
        }, f, indent=2)

if __name__ == '__main__':
    main()
