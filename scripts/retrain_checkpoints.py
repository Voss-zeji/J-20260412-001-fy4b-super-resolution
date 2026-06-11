#!/root/miniconda3/envs/mamba2/bin/python -u
"""Retrain selected methods with checkpoint saving (CH07, 200 epoch, save .pth)"""

import sys, os, json, time, importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
sys.path.insert(0, str(PROJECT_ROOT))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

METHODS = {
    'PFTSR': {
        'dir': PROJECT_ROOT / 'lv1_macro' / 'methods' / '04_method_pftsr',
        'class': 'PFTSR',
        'kwargs': {'in_channels': 1, 'out_channels': 1, 'num_features': 64,
                   'num_pft_blocks': 3, 'num_rb_per_block': 3,
                   'upscale_factor': 2, 'use_attention': True},
        'args': {'epochs': 200, 'batch_size': 16, 'lr': 0.0001},
    },
    'DualScaleRestorer': {
        'dir': PROJECT_ROOT / 'lv2_micro' / 'lv2-save' / '14_method_dualscalerestore',
        'class': 'DualScaleRestorer',
        'kwargs': {'in_channels': 1, 'out_channels': 1, 'upscale_factor': 2},
        'args': {'epochs': 200, 'batch_size': 16, 'lr': 0.0001},
    },
    'DualBranchEmambaIR': {
        'dir': PROJECT_ROOT / 'lv2_micro' / 'methods_new' / '33_method_dual_branch_emambair',
        'class': 'DualBranchEmambaIR',
        'kwargs': {},
        'args': {'epochs': 200, 'batch_size': 8, 'lr': 0.0001},
    },
}


def import_model(method_dir, class_name, kwargs):
    spec = importlib.util.spec_from_file_location('model', method_dir / 'main.py')
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(method_dir))
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        spec.loader.exec_module(mod)
        Cls = getattr(mod, class_name)
        model = Cls(**kwargs).to(DEVICE)
        return model
    finally:
        sys.path = old_path


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = F.l1_loss(sr, hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        sr = torch.clamp(sr, -1, 1)
        for i in range(sr.size(0)):
            total_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
            total_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
            count += 1
    return total_psnr / count, total_ssim / count


def retrain(name):
    cfg = METHODS[name]
    print(f'\n{"="*60}')
    print(f'Retraining: {name}')
    print(f'Device: {DEVICE}')
    print(f'{"="*60}')

    ckpt_path = CHECKPOINT_DIR / f'{name}_CH07_best.pth'
    result_path = CHECKPOINT_DIR / f'{name}_training_result.json'

    # Check if already done
    if ckpt_path.exists():
        if result_path.exists():
            with open(result_path) as f:
                d = json.load(f)
            if d.get('status') == 'success' and d.get('epochs', 0) >= 200:
                psnr = d.get('best_psnr', 0)
                print(f'  Already completed: {name} (PSNR={psnr})')
                return

    model = import_model(cfg['dir'], cfg['class'], cfg['kwargs'])
    nparams = sum(p.numel() for p in model.parameters())
    print(f'Model: {nparams:,} params')

    low_dir = '/root/autodl-tmp/FY-4B/calibration/4000M/CH07'
    high_dir = '/root/autodl-tmp/FY-4B/calibration/2000M/CH07'
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_dir, high_res_dir=high_dir, channel='Channel07',
        batch_size=cfg['args']['batch_size'], num_workers=0,
        patch_size=64, upscale_factor=2, max_samples=100)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['args']['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['args']['epochs'])

    best_psnr, best_ssim, best_epoch = 0, 0, 0
    start_time = time.time()
    total_batches = len(train_loader)

    for epoch in range(1, cfg['args']['epochs'] + 1):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_psnr, val_ssim = validate(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = (time.time() - start_time) / 60

        print(f'Epoch [{epoch}/{cfg["args"]["epochs"]}] '
              f'Loss={loss:.4f} | Val PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} | {elapsed:.0f}min')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f'  >> New best @ epoch {epoch}: {best_psnr:.2f} dB')

    elapsed = time.time() - start_time
    result = {
        'method': name,
        'band': 'CH07',
        'best_psnr': round(best_psnr, 4),
        'best_ssim': round(best_ssim, 4),
        'best_epoch': best_epoch,
        'epochs': cfg['args']['epochs'],
        'params': nparams,
        'status': 'success',
        'runtime_seconds': round(elapsed),
        'checkpoint': str(ckpt_path),
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nDone! Best PSNR={best_psnr:.2f} @ epoch {best_epoch}')
    print(f'Runtime: {elapsed/60:.1f} min')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Result: {result_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', required=True,
                        choices=list(METHODS.keys()))
    parser.add_argument('--log', type=str, default='',
                        help='Log file path')
    args = parser.parse_args()

    if args.log:
        log_f = open(args.log, 'w')
        sys.stdout = log_f
        sys.stderr = log_f

    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    for m in args.methods:
        retrain(m)

    if args.log:
        log_f.close()
