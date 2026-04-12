#!/usr/bin/env python3
"""
compare.py - 统一比较入口

运行所有方法的实验并生成比较结果
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd


def run_method(method_dir, band, epochs, output_file):
    """运行单个方法"""
    main_py = method_dir / "main.py"
    if not main_py.exists():
        print(f"警告: {main_py} 不存在，跳过")
        return None

    method_name = method_dir.name
    print(f"\n{'='*60}")
    print(f"运行: {method_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(main_py),
        "--band", band,
        "--epochs", str(epochs),
        "--output", str(output_file)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        print(result.stdout)
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
            return {"method": method_name, "status": "failed", "error": result.stderr}

        # 读取结果
        with open(output_file, 'r') as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        print(f"超时: {method_name}")
        return {"method": method_name, "status": "timeout"}
    except Exception as e:
        print(f"异常: {e}")
        return {"method": method_name, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description='Compare SR methods')
    parser.add_argument('--level', type=str, default='macro', choices=['macro', 'all'])
    parser.add_argument('--band', type=str, default='CH07', choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--output', type=str, default='lv1_macro/results.csv')
    args = parser.parse_args()

    # 方法目录
    methods_dir = Path('lv1_macro/methods')
    method_dirs = sorted([d for d in methods_dir.iterdir() if d.is_dir()])

    print("="*60)
    print(f"FY4B 超分辨率方法比较")
    print("="*60)
    print(f"Band: {args.band}")
    print(f"Epochs: {args.epochs}")
    print(f"方法数: {len(method_dirs)}")
    print("="*60)

    results = []
    for method_dir in method_dirs:
        output_file = method_dir / "result.json"
        result = run_method(method_dir, args.band, args.epochs, output_file)
        if result:
            results.append(result)

    # 整理结果
    df = pd.DataFrame(results)

    # 确保所有列存在
    required_cols = ['method', 'band', 'val_psnr', 'val_ssim', 'val_rmse',
                     'train_time', 'train_epochs', 'model_params', 'model_size_mb',
                     'inference_time_ms', 'status']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col != 'status' else 'unknown'

    # 按val_psnr排序
    df = df.sort_values('val_psnr', ascending=False)
    df['rank'] = range(1, len(df) + 1)

    # 保存结果
    df.to_csv(args.output, index=False)

    print("\n" + "="*60)
    print("比较完成!")
    print("="*60)
    print(df[['rank', 'method', 'val_psnr', 'val_ssim', 'model_params', 'status']].to_string(index=False))
    print("="*60)
    print(f"结果已保存: {args.output}")

    # 输出最佳方法
    best = df.iloc[0]
    print(f"\n最佳方法: {best['method']}")
    print(f"val_psnr: {best['val_psnr']:.2f} dB")


if __name__ == '__main__':
    main()
