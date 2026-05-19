#!/usr/bin/env python3
"""
lv2-save 批量运行脚本
运行所有适用的超分辨率方法（10-20，共11个）
规则：30分钟超时 或 50 epoch 自动停止
"""

import subprocess
import json
import time
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

# 配置
LV2_SAVE_DIR = Path(__file__).parent
METHODS_DIR = LV2_SAVE_DIR
EXPERIMENTS_DIR = LV2_SAVE_DIR / "experiments"
MAX_EPOCHS = 50
TIMEOUT_MINUTES = 30

PYTHON_BIN = "/root/miniconda3/envs/mamba2/bin/python"

# 11个适用方法
METHODS = [
    "10_method_swinrestorer",
    "11_method_edgepft",
    "12_method_latentswin",
    "13_method_mambapft",
    "14_method_dualscalerestore",
    "15_method_ntire2026_ir_sr",
    "16_method_weather_sr",
    "17_method_emambair",
    "18_method_gprof_ir",
    "19_method_impa_net",
    "20_method_multispectral_sr",
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

    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    shutil.copytree(method_dir, exp_dir)
    exp_main_py = exp_dir / "main.py"

    output_file = exp_dir / "result.json"
    log_file = exp_dir / "training.log"

    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"开始运行: {method_name}")
    print(f"实验目录: {exp_name}")
    print(f"开始时间: {start_timestamp}")
    print(f"计划: 最多 {MAX_EPOCHS} epochs, {TIMEOUT_MINUTES}分钟超时")
    print(f"波段: {band}")
    print(f"{'='*60}")

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

    last_epoch = 0
    last_psnr = 0.0

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

            for line in process.stdout:
                log_f.write(line)
                log_f.flush()

                if 'Epoch' in line:
                    m = re.search(r'Epoch\s*\[(\d+)/(\d+)\]', line)
                    if m:
                        last_epoch = int(m.group(1))
                    m = re.search(r'PSNR[:\s]+([\d.]+)', line, re.I)
                    if m:
                        last_psnr = float(m.group(1))

            process.wait()
            returncode = process.returncode

            if returncode == 124:
                log_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 达到时间限制({TIMEOUT_MINUTES}分钟)\n")
            elif returncode != 0:
                log_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 进程异常退出: {returncode}\n")

    except Exception as e:
        print(f"[错误] 运行 {method_name} 时出错: {e}")
        return None

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
            "epochs": last_epoch,
            "best_psnr": last_psnr if last_psnr > 0 else None,
        }

    result_data["runtime_seconds"] = actual_runtime
    result_data["start_time"] = start_timestamp
    result_data["end_time"] = end_timestamp

    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"结束时间: {end_timestamp}")
    print(f"实际运行: {actual_runtime/60:.1f} 分钟")
    print(f"运行epoch: {result_data.get('epochs', last_epoch)}")
    print(f"截至精度: PSNR={result_data.get('best_psnr', last_psnr):.2f}" if (result_data.get('best_psnr') or last_psnr) else "截至精度: N/A")

    return result_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run lv2-save methods')
    parser.add_argument('--band', type=str, default='CH07', choices=['CH07', 'CH08'])
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--methods', nargs='+', help='指定方法序号，如 10 15 16')
    args = parser.parse_args()

    global MAX_EPOCHS, TIMEOUT_MINUTES
    MAX_EPOCHS = args.max_epochs
    TIMEOUT_MINUTES = args.timeout

    print("="*60)
    print("FY-4B Super Resolution - lv2-save 适用方法批量运行")
    print(f"波段: {args.band}")
    print(f"每个方法: 最多 {MAX_EPOCHS} epochs, {TIMEOUT_MINUTES}分钟超时")
    print("="*60)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # 过滤方法
    if args.methods:
        filtered = [m for m in METHODS if any(m.startswith(f"{n}_") for n in args.methods)]
    else:
        filtered = METHODS

    print(f"待运行方法({len(filtered)}个): {', '.join(filtered)}")

    all_results = []
    for i, method in enumerate(filtered, 1):
        print(f"\n[{i}/{len(filtered)}] 准备运行 {method}...")
        result = run_method(method, args.band)
        if result:
            all_results.append({
                "method": method,
                "status": result.get("status", "unknown"),
                "best_psnr": result.get("best_psnr") or result.get("val_psnr"),
                "best_ssim": result.get("best_ssim") or result.get("val_ssim"),
                "runtime_seconds": result.get("runtime_seconds"),
                "epochs": result.get("epochs", 0),
            })

    # 保存汇总
    summary = {
        "total_methods": len(filtered),
        "completed": len(all_results),
        "band": args.band,
        "max_epochs": MAX_EPOCHS,
        "timeout_minutes": TIMEOUT_MINUTES,
        "results": sorted(all_results, key=lambda x: x.get('best_psnr') or 0, reverse=True),
    }

    summary_file = EXPERIMENTS_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"批量运行完成! 结果已保存到 {summary_file}")
    print(f"{'='*60}")

    print("\n结果排名(按PSNR降序):")
    for i, r in enumerate(summary["results"], 1):
        psnr = r.get('best_psnr')
        psnr_str = f"{psnr:.2f}" if psnr else "N/A"
        ssim = r.get('best_ssim')
        ssim_str = f"{ssim:.4f}" if ssim else "N/A"
        rt = r.get('runtime_seconds', 0)
        print(f"  {i:2d}. {r['method']}: PSNR={psnr_str}, SSIM={ssim_str}, "
              f"Epochs={r.get('epochs',0)}, Runtime={rt/60:.1f}min, Status={r.get('status','N/A')}")


if __name__ == "__main__":
    main()
