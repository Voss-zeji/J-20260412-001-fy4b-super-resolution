#!/root/miniconda3/envs/mamba2/bin/python -u
"""LV3 Fusion: Ensemble + Distillation of Top-2 models"""

import sys, os, json, time, importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
sys.path.insert(0, str(PROJECT_ROOT))
from utils import calculate_psnr, calculate_ssim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM_MIN, NORM_MAX = 150.0, 350.0


def load_model(name, mod_path, class_name, kwargs, ckpt_name):
    ckpt = CHECKPOINT_DIR / ckpt_name
    if not ckpt.exists():
        print(f'[WARN] {name}: checkpoint not found')
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
        nparams = sum(p.numel() for p in model.parameters())
        print(f'  [OK] {name}: {nparams:,} params')
        return model
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        return None
    finally:
        sys.path = old_path


@torch.no_grad()
def ensemble_inference(model1, model2, lr_batch):
    sr1 = model1(lr_batch)
    sr2 = model2(lr_batch)
    return (sr1 + sr2) / 2


def evaluate_ensemble():
    from data.fy4b_dataset import create_dataloaders

    model1 = load_model('EmambaIR', 'lv2_micro/lv2-save/17_method_emambair',
                        'EmambaIRNet', {'in_channels': 1, 'out_channels': 1,
                                        'base_channels': 64, 'num_blocks': 8},
                        'EmambaIR_CH07_best.pth')
    model2 = load_model('SFGSwinIR', 'lv2_micro/methods_new/31_method_sfg_swinir',
                        'SFGSwinIR', {'in_channels': 1, 'out_channels': 1,
                                      'embed_dim': 60, 'depths': [4, 4],
                                      'num_heads': [4, 4], 'window_size': 8,
                                      'mlp_ratio': 2.0, 'upscale_factor': 2},
                        'SFGSwinIR_CH07_best.pth')

    if model1 is None or model2 is None:
        print('[FAIL] Cannot load both models for ensemble')
        return None, None

    _, val_loader = create_dataloaders(
        low_res_dir='/root/autodl-tmp/Calibration-FY4B/4000M/CH07',
        high_res_dir='/root/autodl-tmp/Calibration-FY4B/2000M/CH07',
        channel='Channel07', batch_size=4, num_workers=0,
        patch_size=64, upscale_factor=2, max_samples=50,
    )

    total_psnr, total_ssim, count = 0.0, 0.0, 0
    for lr, hr, _ in val_loader:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        sr = ensemble_inference(model1, model2, lr)
        sr = torch.clamp(sr, -1, 1)
        for i in range(sr.size(0)):
            total_psnr += calculate_psnr(sr[i:i + 1], hr[i:i + 1])
            total_ssim += calculate_ssim(sr[i:i + 1], hr[i:i + 1])
            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f'\nEnsemble (EmambaIR + SFGSwinIR):')
    print(f'  PSNR = {avg_psnr:.2f} dB')
    print(f'  SSIM = {avg_ssim:.4f}')
    print(f'  vs EmambaIR alone: 44.42 dB')
    print(f'  vs SFGSwinIR alone: 44.30 dB')
    print(f'  Improvement: {avg_psnr - 44.42:+.2f} dB')

    return avg_psnr, avg_ssim


def evaluate_individual():
    """Evaluate each model individually for fair comparison"""
    from data.fy4b_dataset import create_dataloaders

    configs = [
        ('EmambaIR', 'lv2_micro/lv2-save/17_method_emambair', 'EmambaIRNet',
         {'in_channels': 1, 'out_channels': 1, 'base_channels': 64, 'num_blocks': 8},
         'EmambaIR_CH07_best.pth'),
        ('SFGSwinIR', 'lv2_micro/methods_new/31_method_sfg_swinir', 'SFGSwinIR',
         {'in_channels': 1, 'out_channels': 1, 'embed_dim': 60,
          'depths': [4, 4], 'num_heads': [4, 4], 'window_size': 8,
          'mlp_ratio': 2.0, 'upscale_factor': 2},
         'SFGSwinIR_CH07_best.pth'),
        ('PFTSR', 'lv1_macro/methods/04_method_pftsr', 'PFTSR',
         {'in_channels': 1, 'out_channels': 1, 'num_features': 64,
          'num_pft_blocks': 3, 'num_rb_per_block': 3, 'upscale_factor': 2, 'use_attention': True},
         'PFTSR_CH07_best.pth'),
        ('DualScaleRestorer', 'lv2_micro/lv2-save/14_method_dualscalerestore', 'DualScaleRestorer',
         {'in_channels': 1, 'out_channels': 1, 'upscale_factor': 2},
         'DualScaleRestorer_CH07_best.pth'),
    ]

    _, val_loader = create_dataloaders(
        low_res_dir='/root/autodl-tmp/Calibration-FY4B/4000M/CH07',
        high_res_dir='/root/autodl-tmp/Calibration-FY4B/2000M/CH07',
        channel='Channel07', batch_size=4, num_workers=0,
        patch_size=64, upscale_factor=2, max_samples=50,
    )

    results = {}
    for name, mpath, cls, kwargs, ckpt in configs:
        model = load_model(name, mpath, cls, kwargs, ckpt)
        if model is None:
            continue
        total_psnr, total_ssim, count = 0.0, 0.0, 0
        for lr, hr, _ in val_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = model(lr)
            sr = torch.clamp(sr, -1, 1)
            for i in range(sr.size(0)):
                total_psnr += calculate_psnr(sr[i:i + 1], hr[i:i + 1])
                total_ssim += calculate_ssim(sr[i:i + 1], hr[i:i + 1])
                count += 1
        results[name] = {'psnr': round(total_psnr / count, 2), 'ssim': round(total_ssim / count, 4)}
        print(f'  {name}: PSNR={results[name]["psnr"]:.2f}, SSIM={results[name]["ssim"]:.4f}')
        del model
        torch.cuda.empty_cache()

    return results


def main():
    print('=' * 60)
    print('LV3 Fusion: Ensemble Evaluation')
    print('=' * 60)

    print('\n--- Individual Model Evaluation ---')
    individual = evaluate_individual()

    print('\n--- Ensemble Evaluation (EmambaIR + SFGSwinIR) ---')
    ensemble_psnr, ensemble_ssim = evaluate_ensemble()

    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    if individual:
        for name, r in individual.items():
            print(f'  {name:<25} PSNR={r["psnr"]:.2f}  SSIM={r["ssim"]:.4f}')
    if ensemble_psnr:
        print(f'  {"Ensemble (EmambaIR+SFGSwinIR)":<25} PSNR={ensemble_psnr:.2f}  SSIM={ensemble_ssim:.4f}')

    result = {
        'individual': individual,
        'ensemble': {
            'method': 'EmambaIR + SFGSwinIR',
            'psnr': round(ensemble_psnr, 2) if ensemble_psnr else None,
            'ssim': round(ensemble_ssim, 4) if ensemble_ssim else None,
        },
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = PROJECT_ROOT / 'lv3_fusion' / 'fusion_result.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResult saved: {out_path}')


if __name__ == '__main__':
    main()
