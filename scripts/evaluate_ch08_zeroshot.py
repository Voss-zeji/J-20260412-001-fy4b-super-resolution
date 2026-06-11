#!/usr/bin/env python3
"""
CH08 zero-shot 评估脚本
用 CH07 训练的 checkpoint 直接在 CH08 上推理，测试跨通道泛化性

使用方法:
    cd /root/jobs/J-20260412-001-fy4b-super-resolution
    /root/miniconda3/envs/mamba2/bin/python scripts/evaluate_ch08_zeroshot.py
"""

import sys, json, time, importlib, os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

MODEL_CONFIGS = {
    'EmambaIR': {
        'path': 'lv2_micro/lv2-save/17_method_emambair',
        'class_name': 'EmambaIRNet',
        'kwargs': {'in_channels': 1, 'out_channels': 1, 'base_channels': 64, 'num_blocks': 8},
    },
    'SFGSwinIR': {
        'path': 'lv2_micro/methods_new/31_method_sfg_swinir',
        'class_name': 'SFGSwinIR',
        'kwargs': {'in_channels': 1, 'out_channels': 1, 'embed_dim': 60,
                   'depths': [4, 4], 'num_heads': [4, 4],
                   'window_size': 8, 'mlp_ratio': 2.0, 'upscale_factor': 2},
    },
    'PhysicsPFTSR': {
        'path': 'lv2_micro/methods_new/32_method_physics_pftsr',
        'class_name': 'PFTSR',
        'kwargs': {},
    },
}


def build_model(name):
    cfg = MODEL_CONFIGS[name]
    ckpt = CHECKPOINT_DIR / f'{name}_CH07_best.pth'
    if not ckpt.exists():
        print(f'  [skip] {ckpt.name} not found')
        return None

    mod_path = PROJECT_ROOT / cfg['path'] / 'main.py'
    spec = importlib.util.spec_from_file_location(f'mod_{name}', mod_path)
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(PROJECT_ROOT / cfg['path']))
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        spec.loader.exec_module(mod)
        Cls = getattr(mod, cfg['class_name'])
        model = Cls(**cfg['kwargs'])
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model = model.to(DEVICE).eval()
        nparams = sum(p.numel() for p in model.parameters())
        print(f'  [OK] {name}: {nparams:,} params')
        return model
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        return None
    finally:
        sys.path = old_path


@torch.no_grad()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    times = []
    for lr, hr, _ in loader:
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)
        t0 = time.time()
        sr = model(lr)
        t1 = time.time()
        times.append((t1 - t0) / lr.size(0) * 1000)
        sr = torch.clamp(sr, -1, 1)
        for i in range(sr.size(0)):
            total_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
            total_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
            count += 1
    return total_psnr / count, total_ssim / count, sum(times)/len(times)


def main():
    print('=' * 60)
    print('CH08 Zero-Shot 评估')
    print(f'设备: {DEVICE}')
    print('=' * 60)

    _, val_loader = create_dataloaders(
        low_res_dir='/root/autodl-tmp/FY-4B/calibration/4000M/CH08',
        high_res_dir='/root/autodl-tmp/FY-4B/calibration/2000M/CH08',
        channel='Channel08', batch_size=4, num_workers=0,
        patch_size=64, upscale_factor=2, max_samples=50,
    )
    print(f'验证集: {len(val_loader.dataset)} 样本\n')

    ch07_ref = {'EmambaIR': 44.42, 'SFGSwinIR': 44.30, 'PhysicsPFTSR': 44.17}
    results = {}

    for name in ['EmambaIR', 'SFGSwinIR', 'PhysicsPFTSR']:
        print(f'\n--- {name} ---')
        model = build_model(name)
        if model is None:
            continue
        try:
            psnr, ssim, t_ms = evaluate(model, val_loader)
            results[name] = {'ch08_psnr': round(psnr,4), 'ch08_ssim': round(ssim,4), 'infer_ms': round(t_ms,2)}
            gap = psnr - ch07_ref[name]
            print(f'  CH08: {psnr:.2f} dB / {ssim:.4f}  |  CH07: {ch07_ref[name]:.2f}  |  差距: {gap:+.2f} dB')
        except Exception as e:
            print(f'  [FAIL] {name}: {e}')
            import traceback; traceback.print_exc()

    print('\n' + '=' * 60)
    print('汇总')
    print('=' * 60)
    print(f'{"方法":<20} {"CH08 PSNR":<12} {"CH08 SSIM":<12} {"CH07 PSNR":<12} {"差距":<10}')
    print('-' * 66)
    for name in ['EmambaIR', 'SFGSwinIR', 'PhysicsPFTSR']:
        r = results.get(name)
        if r:
            gap = r['ch08_psnr'] - ch07_ref[name]
            print(f'{name:<20} {r["ch08_psnr"]:<12.2f} {r["ch08_ssim"]:<12.4f} {ch07_ref[name]:<12.2f} {gap:<+10.2f}')

    out_path = PROJECT_ROOT / 'lv3_fusion' / 'ch08_zeroshot_results.json'
    with open(out_path, 'w') as f:
        json.dump({'band':'CH08','source':'zero-shot','results':results,'generated_at':time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    print(f'\n结果保存: {out_path}')


if __name__ == '__main__':
    main()
