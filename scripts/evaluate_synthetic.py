#!/root/miniconda3/envs/mamba2/bin/python -u
"""
Phase B2: CH07 Synthetic Test Set — Validate 6-model ranking

1. Load CH07 2000M (real HR) → degrade → synthetic 4000M
2. Run all 6 models on synthetic 4000M
3. Compute PSNR vs real 2000M (ground truth)
4. Compare ranking with original CH07 real-pair ranking
5. If ranking consistent → degradation is good for CH08
"""

import sys, os, json, time, importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from degradation_params import degrade
from data.fy4b_dataset import FY4BDataset
from utils import calculate_psnr, calculate_ssim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
DATA_2K = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')
DATA_4K = Path('/root/autodl-tmp/Calibration-FY4B/4000M/CH07')

NORM_MIN, NORM_MAX = 150.0, 350.0

MODELS = [
    ('EmambaIR', 'lv2_micro/lv2-save/17_method_emambair', 'EmambaIRNet',
     {'in_channels': 1, 'out_channels': 1, 'base_channels': 64, 'num_blocks': 8}),
    ('PFTSR', 'lv1_macro/methods/04_method_pftsr', 'PFTSR',
     {'in_channels': 1, 'out_channels': 1, 'num_features': 64,
      'num_pft_blocks': 3, 'num_rb_per_block': 3, 'upscale_factor': 2, 'use_attention': True}),
    ('DualScaleRestorer', 'lv2_micro/lv2-save/14_method_dualscalerestore', 'DualScaleRestorer',
     {'in_channels': 1, 'out_channels': 1, 'upscale_factor': 2}),
    ('SFGSwinIR', 'lv2_micro/methods_new/31_method_sfg_swinir', 'SFGSwinIR',
     {'in_channels': 1, 'out_channels': 1, 'embed_dim': 60,
      'depths': [4, 4], 'num_heads': [4, 4], 'window_size': 8,
      'mlp_ratio': 2.0, 'upscale_factor': 2}),
    ('DualBranchEmambaIR', 'lv2_micro/methods_new/33_method_dual_branch_emambair', 'DualBranchEmambaIR', {}),
    ('PhysicsPFTSR', 'lv2_micro/methods_new/32_method_physics_pftsr', 'PFTSR', {}),
]

# Original real-pair ranking (CH07, 200 epoch)
REAL_RANKING = {
    'EmambaIR': 44.42, 'PFTSR': 44.35, 'DualScaleRestorer': 44.34,
    'SFGSwinIR': 44.30, 'DualBranchEmambaIR': 44.28, 'PhysicsPFTSR': 44.17,
}


def load_model(name, mod_path, class_name, kwargs):
    ckpt = CHECKPOINT_DIR / f'{name}_CH07_best.pth'
    if not ckpt.exists():
        print(f'  [WARN] {name}: checkpoint not found')
        return None

    mod_full = PROJECT_ROOT / mod_path / 'main.py'
    spec = importlib.util.spec_from_file_location(f'mod_{name}', mod_full)
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(PROJECT_ROOT / mod_path))
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        spec.loader.exec_module(mod)
        Cls = getattr(mod, class_name)
        model = Cls(**kwargs).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model
    finally:
        sys.path = old_path


@torch.no_grad()
def infer_patch(model, lr_patch):
    """Single patch inference, return SR in physical space"""
    t = torch.from_numpy(lr_patch[np.newaxis, np.newaxis, ...]).float().to(DEVICE)
    t = (t - NORM_MIN) / (NORM_MAX - NORM_MIN) * 2 - 1
    sr = model(t)
    sr_phys = (sr + 1) / 2 * (NORM_MAX - NORM_MIN) + NORM_MIN
    return sr_phys.squeeze().cpu().numpy()


def patch_inference(model, lr_full):
    """Full-disk patch inference to avoid OOM"""
    H, W = lr_full.shape
    ps = 64
    stride = 56  # overlap=8
    spp = ps * 2  # SR patch size = 128

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
    print('Phase B2: CH07 Synthetic Test Set — Ranking Validation')
    print(f'Device: {DEVICE}')
    print(f'Degradation: PSF σ=0.384, Noise σ=0.4355')
    print('=' * 60)

    # Load synthetic test set
    print('\nLoading CH07 2000M data and degrading...')
    dataset = FY4BDataset(
        low_res_dir=str(DATA_4K), high_res_dir=str(DATA_2K),
        channel='Channel07', mode='val', max_samples=None,
        patch_size=64, upscale_factor=2)
    print(f'Total validation pairs: {len(dataset)}')

    # Use 20 samples for evaluation (center crop for consistency)
    n_test = min(20, len(dataset))
    results = {}

    for name, mod_path, cls_name, kwargs in MODELS:
        print(f'\n--- {name} ---')
        model = load_model(name, mod_path, cls_name, kwargs)
        if model is None:
            continue

        total_psnr, total_ssim, count = 0.0, 0.0, 0
        for idx in range(n_test):
            lr_t, hr_t, info = dataset[idx]
            hr_np = hr_t.squeeze().numpy()  # [-1, 1] normalized, 128x128

            # Convert to physical space for degradation
            hr_phys = (hr_np + 1) / 2 * (NORM_MAX - NORM_MIN) + NORM_MIN
            lr_synth_phys = degrade(hr_phys)

            # Run SR model (normalizes internally)
            sr_phys = patch_inference(model, lr_synth_phys)

            # Crop SR to match HR size
            sr_phys = sr_phys[:hr_phys.shape[0], :hr_phys.shape[1]]

            # Normalize both back to [-1, 1] for PSNR/SSIM
            sr_norm = (sr_phys - NORM_MIN) / (NORM_MAX - NORM_MIN) * 2 - 1
            hr_norm = hr_np  # already normalized

            psnr = calculate_psnr(
                torch.from_numpy(sr_norm[np.newaxis, np.newaxis, ...]).float(),
                torch.from_numpy(hr_norm[np.newaxis, np.newaxis, ...]).float())
            ssim = calculate_ssim(
                torch.from_numpy(sr_norm[np.newaxis, np.newaxis, ...]).float(),
                torch.from_numpy(hr_norm[np.newaxis, np.newaxis, ...]).float())
            total_psnr += psnr
            total_ssim += ssim
            count += 1

            if idx < 3:
                print(f'  Sample {idx}: PSNR={psnr:.2f} SSIM={ssim:.4f}')

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        results[name] = {'synth_psnr': round(avg_psnr, 2), 'synth_ssim': round(avg_ssim, 4)}
        print(f'  >>> {name}: synth_PSNR={avg_psnr:.2f} (real={REAL_RANKING.get(name,0):.2f})')

    # Ranking comparison
    print('\n' + '=' * 60)
    print('Ranking Comparison: Synthetic vs Real')
    print('=' * 60)

    synth_sorted = sorted(results.items(), key=lambda x: x[1]['synth_psnr'], reverse=True)
    real_sorted = sorted(REAL_RANKING.items(), key=lambda x: x[1], reverse=True)

    print(f'{"Rank":<6} {"Synthetic":<25} {"Synth(dB)":<12} {"Real(dB)":<12} {"Match":<10}')
    print('-' * 65)

    real_ranks = {m: i+1 for i, (m, _) in enumerate(real_sorted)}
    match = 0
    for synth_rank, (name, r) in enumerate(synth_sorted, 1):
        rr = real_ranks.get(name, 99)
        is_match = '✓' if synth_rank == rr else f'({rr})'
        if synth_rank == rr:
            match += 1
        print(f'{synth_rank:<6} {name:<25} {r["synth_psnr"]:<12.2f} {REAL_RANKING.get(name, 0):<12.2f} {is_match:<10}')

    match_pct = match / len(results) * 100
    print(f'\nRanking agreement: {match}/{len(results)} ({match_pct:.0f}%)')

    # Save results
    out = {
        'method': 'CH07_synthetic_evaluation',
        'degradation_params': {
            'psf_sigma': 0.384, 'noise_std': 0.4355,
        },
        'real_ranking': REAL_RANKING,
        'synth_ranking': {m: r['synth_psnr'] for m, r in results.items()},
        'ranking_agreement_pct': match_pct,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = PROJECT_ROOT / 'lv3_fusion' / 'synthetic_evaluation.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')

    if match_pct >= 80:
        print('\n>>> Degradation model is VALID for CH08 use')
    else:
        print('\n>>> Degradation model needs iteration (B3)')


if __name__ == '__main__':
    main()
