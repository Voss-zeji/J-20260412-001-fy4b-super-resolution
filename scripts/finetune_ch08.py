#!/root/miniconda3/envs/mamba2/bin/python -u
"""Phase C2: CH08 self-supervised fine-tuning (8km->4km) with CH07 transfer"""

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
DATA_DIR = '/root/autodl-tmp/fy4b_ch08_data/selfsup'


class CH08SelfSupDataset(Dataset):
    """CH08 self-supervised dataset: 8km (LR) -> 4km (HR)"""
    def __init__(self, split='train', patch_size=64):
        self.lr_dir = f'{DATA_DIR}/{split}/lr_8km'
        self.hr_dir = f'{DATA_DIR}/{split}/hr_4km'
        self.files = sorted([f.replace('_lr.npy', '') for f in os.listdir(self.lr_dir) if f.endswith('.npy')])
        self.patch_size = patch_size
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base = self.files[idx]
        lr = np.load(f'{self.lr_dir}/{base}_lr.npy').astype(np.float32)
        hr = np.load(f'{self.hr_dir}/{base}_hr.npy').astype(np.float32)

        # Random crop
        lr_ps = self.patch_size
        hr_ps = lr_ps * 2
        if self.split == 'train':
            h = np.random.randint(0, max(1, lr.shape[0] - lr_ps))
            w = np.random.randint(0, max(1, lr.shape[1] - lr_ps))
        else:
            h = max(0, (lr.shape[0] - lr_ps) // 2)
            w = max(0, (lr.shape[1] - lr_ps) // 2)

        lr_patch = lr[h:h+lr_ps, w:w+lr_ps]
        hr_patch = hr[h*2:h*2+hr_ps, w*2:w*2+hr_ps]

        # Normalize to [-1, 1]
        lr_t = (torch.from_numpy(lr_patch) - NORM_MIN) / (NORM_MAX - NORM_MIN) * 2 - 1
        hr_t = (torch.from_numpy(hr_patch) - NORM_MIN) / (NORM_MAX - NORM_MIN) * 2 - 1

        return lr_t.unsqueeze(0).float(), hr_t.unsqueeze(0).float()


def load_ch07_model():
    """Load EmambaIR model with CH07 pretrained weights"""
    mod_path = PROJECT_ROOT / 'lv2_micro' / 'lv2-save' / '17_method_emambair'
    spec = importlib.util.spec_from_file_location('emod', mod_path / 'main.py')
    mod = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    sys.path.insert(0, str(mod_path))
    sys.path.insert(0, str(PROJECT_ROOT))
    spec.loader.exec_module(mod)
    model = mod.EmambaIRNet(in_channels=1, out_channels=1, base_channels=64, num_blocks=8).to(DEVICE)
    state = torch.load(CHECKPOINT_DIR / 'EmambaIR_CH07_best.pth', map_location=DEVICE)
    model.load_state_dict(state)
    sys.path = old_path
    return model


def freeze_encoder(model):
    """Freeze encoder layers (shallow + mamba blocks), keep decoder trainable"""
    for name, param in model.named_parameters():
        if 'upsample' in name or 'reconstruct' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Frozen: {frozen:,} params, Trainable: {trainable:,} params')
    return model


def train():
    print('=' * 60)
    print('Phase C2: CH08 Self-Supervised Fine-Tuning')
    print(f'Device: {DEVICE}')
    print('=' * 60)

    # Data
    train_ds = CH08SelfSupDataset('train', patch_size=64)
    val_ds = CH08SelfSupDataset('val', patch_size=64)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f'Train: {len(train_ds)} samples, Val: {len(val_ds)} samples')

    # Model
    model = load_ch07_model()
    model = freeze_encoder(model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_psnr = 0
    n_epochs = 50
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for lr, hr in train_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            sr = model(lr)
            loss = F.l1_loss(sr, hr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_psnr, val_ssim, vcount = 0, 0, 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = model(lr)
                sr = torch.clamp(sr, -1, 1)
                for i in range(sr.size(0)):
                    val_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
                    val_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
                    vcount += 1

        val_psnr /= vcount
        val_ssim /= vcount
        scheduler.step()
        elapsed = (time.time() - start) / 60

        print(f'Epoch [{epoch:02d}/{n_epochs}] Loss={total_loss/len(train_loader):.4f} | '
              f'Val PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} | {elapsed:.0f}min')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt_path = CHECKPOINT_DIR / 'CH08_adapt_best.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f'  >> New best PSNR={best_psnr:.2f}, saved')

    elapsed = time.time() - start
    print(f'\nDone! Best PSNR={best_psnr:.2f}, Time={elapsed/60:.1f}min')
    print(f'Checkpoint: {CHECKPOINT_DIR / "CH08_adapt_best.pth"}')


if __name__ == '__main__':
    train()
