#!/usr/bin/env python3
"""
select_best.py - 选择最佳方法进入 lv2_micro

根据 lv1_macro/results.csv 选择最佳方法
"""

import json
import pandas as pd
from pathlib import Path


def main():
    results_file = Path('lv1_macro/results.csv')
    target_file = Path('lv2_micro/TARGET_METHOD')

    if not results_file.exists():
        print(f"错误: {results_file} 不存在")
        print("请先运行: python compare.py")
        return

    # 读取结果
    df = pd.read_csv(results_file)

    # 过滤成功的方法
    df_success = df[df['status'] == 'success']

    if len(df_success) == 0:
        print("错误: 没有成功的方法")
        return

    # 按val_psnr排序
    df_success = df_success.sort_values('val_psnr', ascending=False)

    # 选择最佳方法
    best = df_success.iloc[0]

    print("="*60)
    print("方法比较结果")
    print("="*60)
    print(df_success[['method', 'val_psnr', 'val_ssim', 'model_params', 'train_time']].to_string(index=False))
    print("="*60)

    print(f"\n最佳方法: {best['method']}")
    print(f"  val_psnr: {best['val_psnr']:.2f} dB")
    print(f"  val_ssim: {best['val_ssim']:.4f}")
    print(f"  model_params: {best['model_params']:,}")
    print(f"  train_time: {best['train_time']:.1f}s")

    # 写入目标文件
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, 'w') as f:
        f.write(best['method'])

    print(f"\n已写入 {target_file}: {best['method']}")

    # 检查是否触发 lv3_fusion
    if len(df_success) >= 2:
        top2_gap = df_success.iloc[0]['val_psnr'] - df_success.iloc[1]['val_psnr']
        print(f"\nTop-2 gap: {top2_gap:.2f} dB")
        if top2_gap < 0.5:
            print("提示: Top-2 gap < 0.5 dB，建议尝试 lv3_fusion 融合方法")


if __name__ == '__main__':
    main()
