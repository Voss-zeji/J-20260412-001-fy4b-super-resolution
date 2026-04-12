"""
FY-4B AGRI L1 数据定标演示脚本
演示如何将DN值转换为物理量
"""

import sys
sys.path.insert(0, '/root/codes/sp0301/preprocessing')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无界面模式
import matplotlib.pyplot as plt
from fy4b_calibration import FY4BCalibrator, process_single_file
import os

# 配置
INPUT_FILE = '/root/autodl-tmp/2000M/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250322000000_20250322001459_2000M_V0001.HDF'
OUTPUT_DIR = '/root/codes/sp0301/preprocessing/output'

print("="*70)
print("FY-4B AGRI L1 数据定标演示")
print("="*70)

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 创建定标器并分析文件
print("\n【步骤1】加载文件并分析定标信息")
print("-"*70)

calibrator = FY4BCalibrator(INPUT_FILE)
calibrator.print_calibration_info()

# 2. 对比两种定标方法
print("\n【步骤2】对比定标方法效果")
print("-"*70)

# 选择一个区域进行测试（避免内存过大）
test_region = (2000, 2500, 2000, 2500)  # (row_start, row_end, col_start, col_end)

import h5py
with h5py.File(INPUT_FILE, 'r') as f:
    # 读取测试区域的DN值
    ch1_dn = f['Data/NOMChannel01'][test_region[0]:test_region[1], test_region[2]:test_region[3]]
    ch7_dn = f['Data/NOMChannel07'][test_region[0]:test_region[1], test_region[2]:test_region[3]]

print(f"\n测试区域: 行 {test_region[0]}:{test_region[1]}, 列 {test_region[2]}:{test_region[3]}")
print(f"区域大小: {ch1_dn.shape}")

# Channel01 - 可见光波段 (反射率)
print("\n  Channel01 (VIS 0.47μm) - 反射率:")
ch1_lut = calibrator.calibrate_with_lut('Channel01', ch1_dn)
ch1_linear = calibrator.calibrate_linear('Channel01', ch1_dn)

valid_mask = ~np.isnan(ch1_lut)
if valid_mask.any():
    print(f"    查找表方法 - min: {ch1_lut[valid_mask].min():.4f}, max: {ch1_lut[valid_mask].max():.4f}, mean: {ch1_lut[valid_mask].mean():.4f}")
    print(f"    线性方法   - min: {ch1_linear[valid_mask].min():.4f}, max: {ch1_linear[valid_mask].max():.4f}, mean: {ch1_linear[valid_mask].mean():.4f}")
    print(f"    差异       - max abs diff: {np.nanmax(np.abs(ch1_lut - ch1_linear)):.6f}")

# Channel07 - 红外波段 (亮温)
print("\n  Channel07 (IR 3.90μm) - 亮温:")
ch7_lut = calibrator.calibrate_with_lut('Channel07', ch7_dn)
ch7_linear = calibrator.calibrate_linear('Channel07', ch7_dn)

valid_mask = ~np.isnan(ch7_lut)
if valid_mask.any():
    print(f"    查找表方法 - min: {ch7_lut[valid_mask].min():.2f}K, max: {ch7_lut[valid_mask].max():.2f}K, mean: {ch7_lut[valid_mask].mean():.2f}K")
    print(f"    线性方法   - min: {ch7_linear[valid_mask].min():.2f}K, max: {ch7_linear[valid_mask].max():.2f}K, mean: {ch7_linear[valid_mask].mean():.2f}K")
    print(f"    差异       - max abs diff: {np.nanmax(np.abs(ch7_lut - ch7_linear)):.4f}K")

# 3. 可视化对比
print("\n【步骤3】生成可视化图像")
print("-"*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Channel01 - DN值
im0 = axes[0, 0].imshow(ch1_dn.astype(float), cmap='gray', vmin=0, vmax=5000)
axes[0, 0].set_title('Channel01 - DN Values')
axes[0, 0].set_xlabel('Column')
axes[0, 0].set_ylabel('Row')
plt.colorbar(im0, ax=axes[0, 0], label='DN')

# Channel01 - 查找表定标
im1 = axes[0, 1].imshow(ch1_lut, cmap='gray', vmin=0, vmax=1.0)
axes[0, 1].set_title('Channel01 - LUT Calibration (Reflectance)')
axes[0, 1].set_xlabel('Column')
axes[0, 1].set_ylabel('Row')
plt.colorbar(im1, ax=axes[0, 1], label='Reflectance')

# Channel01 - 线性定标
im2 = axes[0, 2].imshow(ch1_linear, cmap='gray', vmin=0, vmax=1.0)
axes[0, 2].set_title('Channel01 - Linear Calibration (Reflectance)')
axes[0, 2].set_xlabel('Column')
axes[0, 2].set_ylabel('Row')
plt.colorbar(im2, ax=axes[0, 2], label='Reflectance')

# Channel07 - DN值
im3 = axes[1, 0].imshow(ch7_dn.astype(float), cmap='hot', vmin=3000, vmax=8000)
axes[1, 0].set_title('Channel07 - DN Values')
axes[1, 0].set_xlabel('Column')
axes[1, 0].set_ylabel('Row')
plt.colorbar(im3, ax=axes[1, 0], label='DN')

# Channel07 - 查找表定标
im4 = axes[1, 1].imshow(ch7_lut, cmap='hot', vmin=280, vmax=310)
axes[1, 1].set_title('Channel07 - LUT Calibration (Brightness Temp)')
axes[1, 1].set_xlabel('Column')
axes[1, 1].set_ylabel('Row')
plt.colorbar(im4, ax=axes[1, 1], label='Temperature (K)')

# Channel07 - 线性定标
im5 = axes[1, 2].imshow(ch7_linear, cmap='hot', vmin=280, vmax=310)
axes[1, 2].set_title('Channel07 - Linear Calibration (Brightness Temp)')
axes[1, 2].set_xlabel('Column')
axes[1, 2].set_ylabel('Row')
plt.colorbar(im5, ax=axes[1, 2], label='Temperature (K)')

plt.tight_layout()
output_fig = os.path.join(OUTPUT_DIR, 'calibration_comparison.png')
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"  图像已保存: {output_fig}")
plt.close()

# 4. 生成查找表可视化
print("\n【步骤4】分析查找表特性")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 反射率波段查找表
for i, ch in enumerate(['Channel01', 'Channel02']):
    cal_table = calibrator.cal_tables[ch]
    dn_range = np.arange(len(cal_table))
    
    axes[0, i].plot(dn_range, cal_table, 'b-', linewidth=1)
    axes[0, i].set_xlabel('DN Value')
    axes[0, i].set_ylabel('Reflectance')
    axes[0, i].set_title(f'{ch} LUT (Reflectance)')
    axes[0, i].set_xlim(0, 4096)
    axes[0, i].grid(True, alpha=0.3)

# 红外波段查找表
for i, ch in enumerate(['Channel07', 'Channel08']):
    cal_table = calibrator.cal_tables[ch]
    dn_range = np.arange(len(cal_table))
    
    axes[1, i].plot(dn_range, cal_table, 'r-', linewidth=1)
    axes[1, i].set_xlabel('DN Value')
    axes[1, i].set_ylabel('Brightness Temperature (K)')
    axes[1, i].set_title(f'{ch} LUT (Brightness Temperature)')
    axes[1, i].set_xlim(0, 4096)
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
output_fig2 = os.path.join(OUTPUT_DIR, 'lookup_tables.png')
plt.savefig(output_fig2, dpi=150, bbox_inches='tight')
print(f"  图像已保存: {output_fig2}")
plt.close()

# 5. 完整文件定标处理
print("\n【步骤5】完整文件定标处理")
print("-"*70)

# 只处理一个波段作为示例，避免内存过大
print("  选择Channel01进行完整图像定标...")
ch1_full = calibrator.calibrate_with_lut('Channel01')

print(f"  Channel01定标完成!")
print(f"    形状: {ch1_full.shape}")
print(f"    有效值范围: [{np.nanmin(ch1_full):.4f}, {np.nanmax(ch1_full):.4f}]")
print(f"    均值: {np.nanmean(ch1_full):.4f}")
print(f"    填充值比例: {np.isnan(ch1_full).sum() / ch1_full.size * 100:.2f}%")

# 保存完整定标结果
output_cal = os.path.join(OUTPUT_DIR, 'channel01_calibrated.npy')
np.save(output_cal, ch1_full)
print(f"  定标结果已保存: {output_cal}")

# 生成预览图
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
im = ax.imshow(ch1_full, cmap='gray', vmin=0, vmax=1.0)
ax.set_title('Channel01 (VIS 0.47μm) - Full Disk Reflectance')
plt.colorbar(im, ax=ax, label='Reflectance', shrink=0.6)
plt.tight_layout()
output_fig3 = os.path.join(OUTPUT_DIR, 'channel01_full_disk.png')
plt.savefig(output_fig3, dpi=100, bbox_inches='tight')
print(f"  预览图已保存: {output_fig3}")
plt.close()

print("\n" + "="*70)
print("演示完成!")
print(f"输出文件保存在: {OUTPUT_DIR}")
print("="*70)

# 列出输出文件
print("\n生成的文件:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    file_path = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"  - {f} ({size_mb:.2f} MB)")
