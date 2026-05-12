#!/usr/bin/env python3
"""
lv2_micro 批量运行新增的 15 种算法
"""

import subprocess
import json
import time
import sys
import os
import argparse
import shutil
from datetime import datetime
from pathlib import Path

# 配置
METHODS_DIR = Path(__file__).parent / "methods_new"
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
MAX_EPOCHS = 50
TIMEOUT_MINUTES = 30  # 30分钟超时

PYTHON_BIN = "/root/miniconda3/envs/mamba2/bin/python"

# 15 种新算法
NEW_METHODS = [
    "15_method_ntire2026_ir_sr",
    "16_method_weather_sr",
    "17_method_emambair",
    "18_method_gprof_ir",
    "19_method_impa_net",
    "20_method_multispectral_sr",
    "21_method_r2h_diff",
    "22_method_pet_adapter",
    "23_method_normalizing_trajectory",
    "24_method_ctdm",
    "25_method_noisegate",
    "26_method_satsurfgs",
    "27_method_sat3r",
    "28_method_crossview_3d",
    "29_method_infogeo",
]


def run_method(method_name: str, band: str = "CH07"):
    """运行单个方法"""
    method_dir = METHODS_DIR / method_name
    main_py = method_dir / "main.py"

    if not main_py.exists():
        print(f"[错误] {method_name}/main.py 不存在，跳过")
        return None

    # 创建实验目录
    exp_name = f"{method_name}_{band}"
    exp_dir = EXPERIMENTS_DIR / exp_name

    # 如果已存在则删除
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    # 复制方法代码到实验目录
    shutil.copytree(method_dir, exp_dir)
    exp_main_py = exp_dir / "main.py"

    output_file = exp_dir / "result.json"
    log_file = exp_dir / "training.log"

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"开始运行: {method_name}")
    print(f"实验目录: {exp_name}")
    print(f"开始时间: {start_timestamp}")
    print(f"计划: 最多 {MAX_EPOCHS} epochs")
    print(f"时间限制: {TIMEOUT_MINUTES} 分钟")
    print(f"波段: {band}")
    print(f"{'='*60}")

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    timeout_seconds = TIMEOUT_MINUTES * 60
    cmd = [
        'timeout', str(timeout_seconds),
        PYTHON_BIN, '-u',
        str(exp_main_py),
        '--band', band,
        '--epochs', str(MAX_EPOCHS),
        '--batch-size', '8',
        '--output', str(output_file),
    ]

    try:
        with open(log_file, "w") as log_f:
            log_f.write(f"[{start_timestamp}] 开始运行 {method_name}\n")
            log_f.write(f"命令: {' '.join(cmd)}\n")
            log_f.write("="*60 + "\n")
            log_f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            last_epoch = 0
            last_psnr = 0.0

            for line in process.stdout:
                log_f.write(line)
                log_f.flush()

                if 'Epoch' in line:
                    import re
                    m = re.search(r'Epoch\s*\[(\d+)/(\d+)\]', line)
                    if m:
                        last_epoch = int(m.group(1))
                    m = re.search(r'PSNR[:\s]+([\d.]+)', line, re.I)
                    if m:
                        last_psnr = float(m.group(1))

            process.wait()
            returncode = process.returncode

            if returncode == 124:
                log_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 达到时间限制\n")
            elif returncode != 0:
                log_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 进程异常退出: {returncode}\n")

    except Exception as e:
        print(f"[错误] 运行 {method_name} 时出错: {e}")
        return None

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    actual_runtime = (end_time - start_time).total_seconds()

    # 读取结果
    result_data = {}
    if output_file.exists():
        try:
            with open(output_file) as f:
                result_data = json.load(f)
        except:
            pass

    if not result_data:
        result_data = {
            "method": method_name,
            "band": band,
            "status": "partial",
            "train_epochs": last_epoch,
            "val_psnr": last_psnr if last_psnr > 0 else None,
        }

    result_data["runtime_seconds"] = actual_runtime
    result_data["start_time"] = start_timestamp
    result_data["end_time"] = end_timestamp

    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"结束时间: {end_timestamp}")
    print(f"实际运行: {actual_runtime/60:.1f} 分钟")
    print(f"运行epoch: {result_data.get('train_epochs', 'N/A')}")
    print(f"截至精度: {result_data.get('val_psnr', 'N/A')}")

    return result_data


def main():
    parser = argparse.ArgumentParser(description='Run lv2_micro new methods')
    parser.add_argument('--band', type=str, default='CH07', choices=['CH07', 'CH08'])
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--methods', nargs='+', help='指定要运行的方法序号，如 15 16 17')
    args = parser.parse_args()

    global MAX_EPOCHS, TIMEOUT_MINUTES
    MAX_EPOCHS = args.max_epochs
    TIMEOUT_MINUTES = args.timeout

    BAND = args.band

    print("="*60)
    print("FY4B Super Resolution - lv2_micro 新算法批量运行")
    print(f"波段: {BAND}")
    print(f"每个方法: 最多 {MAX_EPOCHS} epochs")
    print(f"总时间限制: {TIMEOUT_MINUTES}分钟/方法")
    print("="*60)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 过滤指定的方法
    if args.methods:
        target_methods = [f"{m:02d}_method_*" for m in args.methods]
        filtered_methods = []
        for pattern in target_methods:
            for method in NEW_METHODS:
                if method.startswith(pattern.replace('*', '')):
                    filtered_methods.append(method)
    else:
        filtered_methods = NEW_METHODS

    all_results = []

    for i, method in enumerate(filtered_methods, 1):
        print(f"\n[{i}/{len(filtered_methods)}] 准备运行 {method}...")

        result = run_method(method, BAND)

        if result:
            all_results.append({
                "method": method,
                "status": result.get("status", "unknown"),
                "val_psnr": result.get("best_psnr"),
                "runtime_seconds": result.get("runtime_seconds"),
                "epochs": result.get("epochs", 0),
            })

    # 保存汇总结果
    summary = {
        "total_methods": len(filtered_methods),
        "completed": len(all_results),
        "results": all_results,
    }

    summary_file = EXPERIMENTS_DIR / "new_methods_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"批量运行完成! 结果已保存到 {summary_file}")
    print(f"{'='*60}")

    # 打印结果摘要
    print("\n结果摘要:")
    for r in sorted(all_results, key=lambda x: x.get('val_psnr') or 0, reverse=True):
        print(f"  {r['method']}: PSNR={r.get('val_psnr', 'N/A')}, Epochs={r.get('epochs', 0)}, Status={r.get('status', 'N/A')}")


if __name__ == "__main__":
    main()
