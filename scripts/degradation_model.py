#!/root/miniconda3/envs/mamba2/bin/python -u
"""
Stage B: Degradation Model Calibration

Calibrate PSF + noise model from CH07 real paired data (2000M ↔ 4000M).
The degradation model is used to create synthetic test sets for CH08 validation.

Method:
  1. Load CH07 2000M (HR) and 4000M (LR) paired data
  2. Search for optimal Gaussian PSF σ:
     2000M → blur(σ) → 2× avg downsample → compare with real 4000M
  3. Fit noise model from residuals
  4. Save calibrated parameters
"""

import sys, os, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import optimize
import h5py

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
DATA_2K = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')
DATA_4K = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH07')
OUTPUT_DIR = PROJECT_ROOT / 'scripts'

NORM_MIN, NORM_MAX = 150.0, 350.0


def load_hdf(filepath, band='CH07'):
    with h5py.File(filepath, 'r') as f:
        data = f[band][()]
    return np.where(np.isfinite(data), data, np.nanmean(data)).astype(np.float32)


def downsample_2x(img):
    """2× average pooling: (H, W) → (H/2, W/2)"""
    H, W = img.shape
    return img.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))


def evaluate_degradation(sigma, hr_imgs, lr_real_imgs):
    """Evaluate Gaussian blur σ on multiple images, return avg RMSE"""
    rmse_total = 0
    count = 0
    for hr, lr_real in zip(hr_imgs, lr_real_imgs):
        blurred = gaussian_filter(hr, sigma=sigma, mode='reflect')
        lr_synth = downsample_2x(blurred)
        rmse = np.sqrt(np.mean((lr_synth - lr_real) ** 2))
        rmse_total += rmse
        count += 1
    return rmse_total / count


def calibrate_psf(hr_imgs, lr_imgs):
    """Search optimal Gaussian PSF σ"""
    print('Calibrating PSF (Gaussian σ)...')
    print(f'Using {len(hr_imgs)} image pairs')

    # Coarse search
    sigmas = np.arange(0.1, 3.0, 0.1)
    rmses = []
    for s in sigmas:
        rmse = evaluate_degradation(s, hr_imgs, lr_imgs)
        rmses.append(rmse)
        print(f'  σ={s:.1f}: RMSE={rmse:.4f}')

    best_idx = np.argmin(rmses)
    best_sigma = sigmas[best_idx]
    print(f'Coarse optimal σ={best_sigma:.1f} (RMSE={rmses[best_idx]:.4f})')

    # Fine search around best
    lo = max(0.01, best_sigma - 0.3)
    hi = min(3.0, best_sigma + 0.3)
    fine_sigmas = np.linspace(lo, hi, 20)
    fine_rmses = [evaluate_degradation(s, hr_imgs, lr_imgs) for s in fine_sigmas]
    best_fine = fine_sigmas[np.argmin(fine_rmses)]
    best_fine_rmse = min(fine_rmses)
    print(f'Fine optimal σ={best_fine:.3f} (RMSE={best_fine_rmse:.4f})')

    return best_fine, best_fine_rmse, (sigmas, rmses)


def analyze_residuals(hr_imgs, lr_imgs, sigma_psf):
    """With optimal PSF, analyze residuals to fit noise model"""
    print(f'\nAnalyzing residuals (σ_psf={sigma_psf:.3f})...')

    all_residuals = []
    for hr, lr_real in zip(hr_imgs, lr_imgs):
        blurred = gaussian_filter(hr, sigma=sigma_psf, mode='reflect')
        lr_synth = downsample_2x(blurred)
        resid = lr_real - lr_synth
        all_residuals.append(resid)

    residuals = np.concatenate([r.ravel() for r in all_residuals])

    # Center crop to avoid edge effects
    center_residuals = np.concatenate([
        r[200:2500, 200:2500].ravel() for r in all_residuals
    ])

    print(f'  Total pixels: {len(residuals):,}')
    print(f'  Residual mean: {np.mean(center_residuals):.4f}')
    print(f'  Residual std:  {np.std(center_residuals):.4f}')
    print(f'  Residual P1:   {np.percentile(center_residuals, 1):.4f}')
    print(f'  Residual P99:  {np.percentile(center_residuals, 99):.4f}')
    print(f'  Residual MAD:  {np.median(np.abs(center_residuals)):.4f}')

    # Test normality
    from scipy.stats import norm, kurtosis, skew
    print(f'  Skewness: {skew(center_residuals):.4f}')
    print(f'  Kurtosis: {kurtosis(center_residuals):.4f}')

    noise_model = {
        'type': 'gaussian',
        'mean': float(np.mean(center_residuals)),
        'std': float(np.std(center_residuals)),
        'mad': float(np.median(np.abs(center_residuals))),
    }

    return noise_model, center_residuals


def main():
    print('=' * 60)
    print('CH07 Degradation Model Calibration')
    print('=' * 60)

    # Find paired files (same filename = same timestamp)
    files_2k = sorted([f for f in os.listdir(DATA_2K) if f.endswith('.HDF')])
    files_4k = sorted([f for f in os.listdir(DATA_4K) if f.endswith('.HDF')])
    names_2k = set(os.path.splitext(f)[0] for f in files_2k)
    names_4k = set(os.path.splitext(f)[0] for f in files_4k)
    common_names = sorted(names_2k & names_4k)
    print(f'Paired files: {len(common_names)}')

    # Use 5 pairs for calibration
    n_pairs = min(5, len(common_names))
    sample_names = common_names[:n_pairs]

    hr_imgs, lr_imgs = [], []
    for name in sample_names:
        hr = load_hdf(str(DATA_2K / f'{name}.HDF'), 'CH07')
        lr = load_hdf(str(DATA_4K / f'{name}.HDF'), 'CH07')
        hr_imgs.append(hr)
        lr_imgs.append(lr)
        print(f'  Loaded: {name} (HR={hr.shape}, LR={lr.shape})')

    # Step 1: Calibrate PSF
    print('\n' + '-' * 60)
    sigma_psf, best_rmse, search_data = calibrate_psf(hr_imgs, lr_imgs)

    # Step 2: Analyze residuals
    noise_model, residuals = analyze_residuals(hr_imgs, lr_imgs, sigma_psf)

    # Step 3: Bicubic baseline comparison
    print('\n' + '-' * 60)
    print('Bicubic baseline comparison:')
    from scipy.ndimage import zoom
    bicubic_rmses = []
    for hr, lr_real in zip(hr_imgs, lr_imgs):
        lr_bicubic = zoom(hr, 0.5, order=3)
        rmse = np.sqrt(np.mean((lr_bicubic - lr_real) ** 2))
        bicubic_rmses.append(rmse)
    print(f'  Bicubic downsample avg RMSE: {np.mean(bicubic_rmses):.4f}')
    print(f'  Optimized PSF avg RMSE:     {best_rmse:.4f}')
    improvement = (np.mean(bicubic_rmses) - best_rmse) / np.mean(bicubic_rmses) * 100
    print(f'  Improvement over bicubic:    {improvement:.1f}%')

    # Save calibration result
    result = {
        'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_pairs': n_pairs,
        'psf': {
            'type': 'gaussian',
            'sigma': round(sigma_psf, 4),
            'kernel_size': int(2 * int(3 * sigma_psf) + 1),
        },
        'noise': noise_model,
        'degradation_pipeline': [
            {'step': 1, 'operation': f'gaussian_blur(sigma={sigma_psf:.3f})'},
            {'step': 2, 'operation': 'downsample_2x(average_pool)'},
            {'step': 3, 'operation': f'add_gaussian_noise(std={noise_model["std"]:.4f})'},
        ],
        'bicubic_baseline_rmse': round(float(np.mean(bicubic_rmses)), 4),
        'optimized_rmse': round(best_rmse, 4),
        'improvement_pct': round(improvement, 1),
    }

    out_path = OUTPUT_DIR / 'degradation_params.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved degradation parameters: {out_path}')

    # Also save as Python for easy import
    py_path = OUTPUT_DIR / 'degradation_params.py'
    with open(py_path, 'w') as f:
        f.write(f'''# Auto-generated degradation parameters
# Calibrated: {result['calibrated_at']}

PSF_SIGMA = {sigma_psf:.4f}
NOISE_STD = {noise_model['std']:.4f}
DEGRADATION_PIPELINE = {json.dumps(result['degradation_pipeline'], indent=2)}

def degrade(hr_img):
    """Apply calibrated degradation to HR image to get synthetic LR"""
    from scipy.ndimage import gaussian_filter
    import numpy as np
    blurred = gaussian_filter(hr_img, sigma=PSF_SIGMA, mode='reflect')
    H, W = blurred.shape
    lr = blurred.reshape(H//2, 2, W//2, 2).mean(axis=(1, 3))
    noise = np.random.normal(0, NOISE_STD, lr.shape)
    return lr + noise
''')
    print(f'Saved Python module: {py_path}')

    print('\n' + '=' * 60)
    print('Calibration complete!')
    print(f'PSF σ = {sigma_psf:.3f} px')
    print(f'Noise σ = {noise_model["std"]:.4f} K')
    print('=' * 60)


if __name__ == '__main__':
    main()
