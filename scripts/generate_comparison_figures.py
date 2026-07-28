#!/root/miniconda3/envs/mamba2/bin/python -u
"""Generate comprehensive comparison figures for FY-4B SR results"""

import sys, os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
CH07_PROD = Path('/root/autodl-tmp/products/ch07')
CH08_PROD = Path('/root/autodl-tmp/products/ch08')
OUTPUT_DIR = Path('/root/autodl-tmp/products/comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORM_MIN, NORM_MAX = 150.0, 350.0


def load_npy(path):
    return np.load(path).astype(np.float32)


def fig_ch07_comparison():
    """CH07: SR vs Bicubic vs HR for 3 models"""
    models = ['EmambaIR', 'SFGSwinIR', 'PhysicsPFTSR']
    sample_file = 'FY4B_CH07_CAL_20250301000000'

    import h5py
    hr_dir = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')
    hr_file = sorted(hr_dir.glob(f'{sample_file}*.HDF'))
    hr = None
    if hr_file:
        with h5py.File(hr_file[0], 'r') as f:
            hr = f['CH07'][()].astype(np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for i, model in enumerate(models):
        sr_file = CH07_PROD / model / f'{sample_file}_SR.npy'
        sr = load_npy(sr_file) if sr_file.exists() else None

        if sr is not None:
            axes[0, i].imshow(sr, cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
            axes[0, i].set_title(f'{model} SR')
            axes[0, i].axis('off')
            h, w = sr.shape
            cy, cx = h // 2, w // 2
            axes[1, i].imshow(sr[cy - 300:cy + 300, cx - 300:cx + 300], cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
            axes[1, i].set_title(f'{model} SR (zoom)')
            axes[1, i].axis('off')

    if hr is not None:
        axes[0, 3].imshow(hr, cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        axes[0, 3].set_title('CH07 HR (2000M)')
        axes[0, 3].axis('off')
        h, w = hr.shape
        cy, cx = h // 2, w // 2
        axes[1, 3].imshow(hr[cy - 300:cy + 300, cx - 300:cx + 300], cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        axes[1, 3].set_title('CH07 HR (zoom)')
        axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ch07_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] CH07 comparison saved')


def fig_ch08_comparison():
    """CH08: SR vs Bicubic comparison"""
    import h5py
    sr_file = CH08_PROD / 'FY4B_CH08_CAL_20250301000000_SR.npy'
    sr = load_npy(sr_file) if sr_file.exists() else None

    bicubic_dir = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH08')
    bicubic_file = sorted(bicubic_dir.glob('FY4B_CH08_CAL_20250301000000*.HDF'))
    bicubic = None
    if bicubic_file:
        with h5py.File(bicubic_file[0], 'r') as f:
            bicubic = f['CH08'][()].astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, (data, title) in enumerate([
        (sr, 'CH08 SR (EmambaIR adapted)'),
        (bicubic, 'CH08 Bicubic (2000M)'),
        (sr - bicubic if sr is not None and bicubic is not None else None, 'SR - Bicubic (diff)')
    ]):
        if data is None:
            continue
        if i < 2:
            axes[0, i].imshow(data, cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        else:
            vlim = max(abs(data.min()), abs(data.max()))
            axes[0, i].imshow(data, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

        h, w = data.shape
        cy, cx = h // 2, w // 2
        if i < 2:
            axes[1, i].imshow(data[cy - 300:cy + 300, cx - 300:cx + 300], cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        else:
            axes[1, i].imshow(data[cy - 300:cy + 300, cx - 300:cx + 300], cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axes[1, i].set_title(f'{title} (zoom)')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ch08_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] CH08 comparison saved')


def fig_degradation_validation():
    """Degradation model validation: real LR vs synthetic LR"""
    import h5py
    from scipy.ndimage import gaussian_filter

    hr_dir = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')
    lr_dir = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH07')

    common = sorted(set(f.stem for f in hr_dir.glob('*.HDF')) & set(f.stem for f in lr_dir.glob('*.HDF')))
    if not common:
        print('[SKIP] No paired files for degradation validation')
        return

    name = common[0]
    with h5py.File(hr_dir / f'{name}.HDF', 'r') as f:
        hr = f['CH07'][()].astype(np.float32)
    with h5py.File(lr_dir / f'{name}.HDF', 'r') as f:
        lr_real = f['CH07'][()].astype(np.float32)

    sigma_psf = 0.384
    blurred = gaussian_filter(hr, sigma=sigma_psf, mode='reflect')
    lr_synth = blurred.reshape(hr.shape[0] // 2, 2, hr.shape[1] // 2, 2).mean(axis=(1, 3))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, (data, title) in enumerate([
        (lr_real, 'Real LR (4000M)'),
        (lr_synth, 'Synthetic LR (degraded)'),
        (lr_real - lr_synth, 'Real - Synthetic (residual)')
    ]):
        if i < 2:
            axes[0, i].imshow(data, cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        else:
            vlim = max(abs(data.min()), abs(data.max()))
            axes[0, i].imshow(data, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        if i < 2:
            axes[1, i].imshow(data[cy - 200:cy + 200, cx - 200:cx + 200], cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        else:
            axes[1, i].imshow(data[cy - 200:cy + 200, cx - 200:cx + 200], cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axes[1, i].set_title(f'{title} (zoom)')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'degradation_validation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] Degradation validation saved')


def fig_psnr_ranking():
    """PSNR ranking bar chart: CH07 real vs synthetic"""
    real = {'EmambaIR': 44.42, 'PFTSR': 44.35, 'DualScaleRestorer': 44.34,
            'SFGSwinIR': 44.30, 'DualBranchEmambaIR': 44.28, 'PhysicsPFTSR': 44.17}
    synth = {k: 5.56 for k in real}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(real.keys())
    real_vals = [real[n] for n in names]
    synth_vals = [synth[n] for n in names]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))

    axes[0].barh(names[::-1], real_vals[::-1], color=colors[::-1])
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_title('CH07 Real Paired Validation')
    for i, v in enumerate(real_vals[::-1]):
        axes[0].text(v + 0.1, i, f'{v:.2f}', va='center')

    axes[1].barh(names[::-1], synth_vals[::-1], color=colors[::-1])
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_title('CH07 Synthetic Validation')
    axes[1].set_xlim(0, 10)
    for i, v in enumerate(synth_vals[::-1]):
        axes[1].text(v + 0.1, i, f'{v:.2f}', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'psnr_ranking.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] PSNR ranking chart saved')


def fig_ch08_proxy_psnr():
    """CH08 proxy PSNR: SR vs bicubic across 10 samples"""
    results_file = CH08_PROD / 'results.json'
    if not results_file.exists():
        print('[SKIP] CH08 results.json not found')
        return

    with open(results_file) as f:
        data = json.load(f)

    samples = [r['file'][:20] for r in data['results']]
    sr_psnr = [r['proxy_psnr_sr'] for r in data['results']]
    bicubic_psnr = [r['proxy_psnr_bicubic'] for r in data['results']]

    x = np.arange(len(samples))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, sr_psnr, width, label='SR (adapted)', color='#2196F3')
    ax.bar(x + width / 2, bicubic_psnr, width, label='Bicubic (interpolated)', color='#FF9800')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Proxy PSNR (dB)')
    ax.set_title('CH08 SR Proxy PSNR vs Bicubic Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ch08_proxy_psnr.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] CH08 proxy PSNR chart saved')


def main():
    print('=' * 60)
    print('FY-4B SR Comprehensive Comparison Figures')
    print('=' * 60)

    fig_ch07_comparison()
    fig_ch08_comparison()
    fig_degradation_validation()
    fig_psnr_ranking()
    fig_ch08_proxy_psnr()

    print('\n' + '=' * 60)
    print(f'All figures saved to {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
