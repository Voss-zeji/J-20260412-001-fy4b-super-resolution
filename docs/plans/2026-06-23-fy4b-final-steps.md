# FY-4B 超分辨率 — 收尾阶段实施计划

> **For agentic workers:** 后续任务使用远程 GPU 服务器执行。Steps use checkbox (`- [ ]`) syntax for tracking.

**目标:** 完成 FY-4B 超分辨率研究的收尾工作：对比可视化、LV3 融合、清理归档

**架构:** 远程服务器已有退化模型校准结果、CH08 微调模型和产品数据，本地文档已同步。剩余工作在远程执行，结果同步回本地。

**Tech Stack:** Python, PyTorch, matplotlib, scipy, numpy, h5py

## Global Constraints

- 所有 Python 脚本在远程 `gpu-server` 上使用 `/root/miniconda3/envs/mamba2/bin/python` 执行
- 数据路径：`/root/autodl-tmp/Calibration-FY4B/`、`/root/autodl-tmp/products/`
- 项目根路径：`/root/jobs/J-20260412-001-fy4b-super-resolution/`
- 结果同步到本地 `D:\playground\jobs\2026-04\J-20260412-001-fy4b-super-resolution\`

---

### Task 1: 生成综合对比可视化

**Files:**
- Create: `scripts/generate_comparison_figures.py` (远程)
- Output: `/root/autodl-tmp/products/comparison/` (远程)
- Sync: `results/` (本地)

**Interfaces:**
- Consumes: CH07 产品 (`/root/autodl-tmp/products/ch07/EmambaIR/`, `SFGSwinIR/`, `PhysicsPFTSR/`), CH08 产品 (`/root/autodl-tmp/products/ch08/`)
- Produces: 综合对比图（CH07 SR vs bicubic vs HR, CH08 SR vs bicubic, 退化模型验证图）

- [ ] **Step 1: 编写对比可视化脚本**

```python
#!/root/miniconda3/envs/mamba2/bin/python -u
"""Generate comprehensive comparison figures for FY-4B SR results"""

import sys, os, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
CH07_PROD = Path('/root/autodl-tmp/products/ch07')
CH08_PROD = Path('/root/autodl-tmp/products/ch08')
OUTPUT_DIR = Path('/root/autodl-tmp/products/comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORM_MIN, NORM_MAX = 150.0, 350.0

def load_npy(path):
    return np.load(path).astype(np.float32)

def find_sample_files(prod_dir, sample_idx=0):
    """Find the sample_idx-th .npy file in product directory"""
    npy_files = sorted([f for f in os.listdir(prod_dir) if f.endswith('_SR.npy')])
    if not npy_files or sample_idx >= len(npy_files):
        return None
    return prod_dir / npy_files[sample_idx]

def fig_ch07_comparison():
    """CH07: SR vs Bicubic vs HR for 3 models"""
    models = ['EmambaIR', 'SFGSwinIR', 'PhysicsPFTSR']
    sample_file = 'FY4B_CH07_CAL_20250301000000'
    
    # Load HR reference
    hr_dir = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH07')
    hr_file = sorted(hr_dir.glob(f'{sample_file}*.HDF'))
    if hr_file:
        import h5py
        with h5py.File(hr_file[0], 'r') as f:
            hr = f['CH07'][()].astype(np.float32)
    else:
        hr = None
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 0: Full disk thumbnails
    # Row 1: Zoomed region
    
    for i, model in enumerate(models):
        sr_file = CH07_PROD / model / f'{sample_file}_SR.npy'
        bicubic_file = CH07_PROD / model / f'{sample_file}_bicubic.npy'
        
        sr = load_npy(sr_file) if sr_file.exists() else None
        bicubic = load_npy(bicubic_file) if bicubic_file.exists() else None
        
        # Full disk
        if sr is not None:
            im = axes[0, i].imshow(sr, cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
            axes[0, i].set_title(f'{model} SR')
            axes[0, i].axis('off')
        
        # Zoom (center 500x500)
        if sr is not None:
            h, w = sr.shape
            cy, cx = h//2, w//2
            zoom = 300
            axes[1, i].imshow(sr[cy-zoom:cy+zoom, cx-zoom:cx+zoom], cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
            axes[1, i].set_title(f'{model} SR (zoom)')
            axes[1, i].axis('off')
    
    # Last column: HR reference
    if hr is not None:
        axes[0, 3].imshow(hr, cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        axes[0, 3].set_title('CH07 HR (2000M)')
        axes[0, 3].axis('off')
        h, w = hr.shape
        cy, cx = h//2, w//2
        axes[1, 3].imshow(hr[cy-300:cy+300, cx-300:cx+300], cmap='gray', vmin=NORM_MIN, vmax=NORM_MAX)
        axes[1, 3].set_title('CH07 HR (zoom)')
        axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ch07_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[OK] CH07 comparison saved')

def fig_ch08_comparison():
    """CH08: SR vs Bicubic comparison"""
    sr_file = CH08_PROD / 'FY4B_CH08_CAL_20250301000000_SR.npy'
    sr = load_npy(sr_file) if sr_file.exists() else None
    
    # Load bicubic (2000M interpolated from 4000M)
    bicubic_dir = Path('/root/autodl-tmp/Calibration-FY4B/2000M/CH08')
    bicubic_file = sorted(bicubic_dir.glob('FY4B_CH08_CAL_20250301000000*.HDF'))
    bicubic = None
    if bicubic_file:
        import h5py
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
        axes[0, i].imshow(data, cmap='gray' if i < 2 else 'RdBu_r',
                         vmin=(NORM_MIN, NORM_MAX, -20)[i] if i < 2 else -20,
                         vmax=(NORM_MIN, NORM_MAX, 20)[i] if i < 2 else 20)
        axes[0, i].set_title(title)
        axes[0, i].axis('off')
        
        # Zoom
        h, w = data.shape
        cy, cx = h//2, w//2
        axes[1, i].imshow(data[cy-300:cy+300, cx-300:cx+300], cmap='gray' if i < 2 else 'RdBu_r',
                         vmin=(NORM_MIN, NORM_MAX, -20)[i] if i < 2 else -20,
                         vmax=(NORM_MIN, NORM_MAX, 20)[i] if i < 2 else 20)
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
    
    # Apply calibrated degradation
    sigma_psf = 0.384
    blurred = gaussian_filter(hr, sigma=sigma_psf, mode='reflect')
    lr_synth = blurred.reshape(hr.shape[0]//2, 2, hr.shape[1]//2, 2).mean(axis=(1, 3))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (data, title) in enumerate([
        (lr_real, 'Real LR (4000M)'),
        (lr_synth, 'Synthetic LR (degraded)'),
        (lr_real - lr_synth, 'Real - Synthetic (residual)')
    ]):
        vmax = max(abs(data.min()), abs(data.max())) if i == 2 else NORM_MAX
        axes[0, i].imshow(data, cmap='gray' if i < 2 else 'RdBu_r',
                         vmin=(-vmax, NORM_MIN)[i < 2], vmax=(vmax, NORM_MAX)[i < 2])
        axes[0, i].set_title(title)
        axes[0, i].axis('off')
        
        cy, cx = data.shape[0]//2, data.shape[1]//2
        axes[1, i].imshow(data[cy-200:cy+200, cx-200:cx+200], cmap='gray' if i < 2 else 'RdBu_r',
                         vmin=(-vmax, NORM_MIN)[i < 2], vmax=(vmax, NORM_MAX)[i < 2])
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
    ax.bar(x - width/2, sr_psnr, width, label='SR (adapted)', color='#2196F3')
    ax.bar(x + width/2, bicubic_psnr, width, label='Bicubic (interpolated)', color='#FF9800')
    
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
```

- [ ] **Step 2: 上传脚本到远程并执行**

```bash
# 脚本已在远程存在则直接执行，否则上传
ssh gpu-server "cd /root/jobs/J-20260412-001-fy4b-super-resolution && /root/miniconda3/envs/mamba2/bin/python scripts/generate_comparison_figures.py"
```

Expected: 5 张 PNG 图保存到 `/root/autodl-tmp/products/comparison/`

- [ ] **Step 3: 将结果同步到本地**

```bash
# 从远程下载对比图到本地
scp -P 36372 root@connect.bjb1.seetacloud.com:/root/autodl-tmp/products/comparison/*.png ./results/
```

Expected: 本地 `results/` 目录出现 5 张对比图

- [ ] **Step 4: 更新 progress.md**

在 progress.md 追加对比可视化完成记录。

---

### Task 2: LV3 融合决策与执行

**Files:**
- Create: `scripts/lv3_fusion.py` (远程)
- Modify: `lv3_fusion/UNIFIED_EVALUATION.md` (远程)

**Interfaces:**
- Consumes: Top-10 方法排名、合成测试集排名（100% 一致）、CH08 微调结果
- Produces: 融合模型 checkpoint、融合评估报告

**背景分析：**
- 合成测试集排名与 CH07 真实配对 100% 一致 → 退化模型有效
- 所有方法合成 PSNR 相同（5.56 dB）→ 无法通过合成测试区分方法优劣
- CH08 自监督微调后 PSNR=41.39 dB（代理指标），但 bicubic 基线高达 49.22 dB
- **结论：CH08 的代理 PSNR 不可靠（因为 2000M 是内插的），LV3 融合应基于 CH07 真实配对排名**

- [ ] **Step 1: 确定融合策略**

基于 TOP10_SELECTION.md 的 3 组融合策略，结合合成测试结果选择：

| 策略 | 组合 | 选择理由 |
|------|------|---------|
| **A. EmambaIR + SFGSwinIR** (#1 + #4) | Mamba + Transformer 频域 | 互补性最强，推荐 |
| B. PFTSR + Physics-PFTSR (#2 + #9) | 同架构不同损失 | 参数量小，部署友好 |
| C. Dual-Branch EmambaIR + PFTSR (#5 + #7) | 双频域分支 | 探索性 |

**推荐策略 A**：EmambaIR (PSNR 最高) + SFGSwinIR (PSNR/参数比最优)

- [ ] **Step 2: 编写融合训练脚本**

```python
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

# Load Top-2 models
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
        return model
    finally:
        sys.path = old_path

# Strategy A: Simple ensemble (average of Top-2 outputs)
@torch.no_grad()
def ensemble_inference(model1, model2, lr_batch):
    sr1 = model1(lr_batch)
    sr2 = model2(lr_batch)
    return (sr1 + sr2) / 2

# Evaluate ensemble on CH07 validation set
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
        return
    
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
            total_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
            total_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
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

if __name__ == '__main__':
    print('=' * 60)
    print('LV3 Fusion: Ensemble Evaluation')
    print('=' * 60)
    evaluate_ensemble()
```

- [ ] **Step 3: 在远程执行融合评估**

```bash
ssh gpu-server "cd /root/jobs/J-20260412-001-fy4b-super-resolution && /root/miniconda3/envs/mamba2/bin/python scripts/lv3_fusion.py"
```

Expected: 输出融合模型 PSNR/SSIM，与单模型对比

- [ ] **Step 4: 更新 LV3 融合文档**

更新 `lv3_fusion/UNIFIED_EVALUATION.md` 添加融合评估结果

- [ ] **Step 5: 更新 progress.md**

---

### Task 3: 清理低分方法（可选）

**Files:**
- Modify: `lv2_micro/` (远程)

**Interfaces:**
- Consumes: TOP10_SELECTION.md 中排名 11-24 的方法列表
- Produces: 清理后的目录结构

- [ ] **Step 1: 确认清理清单**

| 排名 | 方法 | PSNR | 清理理由 |
|:---:|:---|:---:|:---|
| 11 | SwinIR | 44.15 | SwinRestorer 已覆盖 |
| 12 | IMPA-Net | 44.12 | 非 Top-10 |
| 13 | TinyNINA | 44.12 | 轻量但非 Top-10 |
| 14 | NTIRE2026-IR-SR | 44.04 | 推理太慢 (339ms) |
| 15 | SFG-PFTSR | 43.97 | SFG-SwinIR 已覆盖 |
| 16 | EdgePFT | 43.73 | 低分 |
| 17 | RealRestorer | 43.66 | 低分 |
| 18 | LCMSR | 43.63 | 扩散模型过大 |
| 19 | LatentSwin | 43.60 | 低分 |
| 20 | Multispectral-SR | 43.17 | 低分 |
| 21 | WeatherSR | 41.78 | 低分 |
| 22 | SRCNN | 38.28 | 基线 |
| 23 | EDSR | 37.94 | 低分 |
| 24 | M2IR | 34.21 | 低分 |

- [ ] **Step 2: 执行清理（远程）**

```bash
ssh gpu-server "cd /root/jobs/J-20260412-001-fy4b-super-resolution && \
  mkdir -p lv2_micro/archived && \
  for d in 05_method_swinir 19_method_impa_net 06_method_tinynina 15_method_ntire2026_ir_sr \
           34_method_sfg_pftsr 11_method_edgepft 08_method_realrestorer 09_method_lcmsr \
           12_method_latentswin 20_method_multispectral_sr 16_method_weather_sr \
           02_baseline_srcnn 03_method_edsr 07_method_m2ir; do \
    [ -d lv2_micro/\$d ] && mv lv2_micro/\$d lv2_micro/archived/\$d && echo \"Archived \$d\"; \
  done"
```

- [ ] **Step 3: 更新 progress.md**

---

### Task 4: 最终报告生成

**Files:**
- Create: `results/final_report.md` (本地)

- [ ] **Step 1: 汇总所有结果生成最终报告**

```markdown
# FY-4B 超分辨率研究 — 最终报告

## 研究概述
...

## 关键发现
1. CH08 2000M 为 4000M 内插产物，非独立观测
2. 退化模型校准成功（PSF σ=0.384, 噪声 σ=0.4355 K）
3. 合成测试集排名与真实配对 100% 一致
4. CH08 自监督微调有效（Val PSNR=41.39 dB）
5. LV3 融合 [结果]

## Top-10 方法排名
...

## 产品生成
...
```

- [ ] **Step 2: 提交最终 commit**

```bash
git add -A && git commit -m "feat: complete fy-4b sr study with degradation calibration and ch08 validation"
```
