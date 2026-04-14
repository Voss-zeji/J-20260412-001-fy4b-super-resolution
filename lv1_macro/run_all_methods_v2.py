#!/usr/bin/env python3
"""
依次运行 lv1_macro 下所有方法，每个方法运行固定步数（而非固定时间）
记录：开始时间、运行步数、截至时的精度
"""

import subprocess
import json
import time
import sys
import os
from datetime import datetime
from pathlib import Path

# 配置
METHODS_DIR = Path(__file__).parent / "methods"
RESULTS_DIR = Path(__file__).parent / "results"
MAX_STEPS_PER_EPOCH = 1000  # 每个epoch最多1000步（实际由数据量决定）
MAX_EPOCHS = 50  # 最多50个epoch
BAND = "CH07"

# 方法列表（按顺序）
METHODS = [
    "01_baseline_bicubic",
    "02_baseline_srcnn",
    "03_method_edsr",
    "04_method_pftsr",
    "05_method_swinir",
    "06_method_tinynina",
    "07_method_m2ir",
    "08_method_realrestorer",
    "09_method_lcmsr",
]

def run_method(method_name: str):
    """运行单个方法"""
    method_dir = METHODS_DIR / method_name
    main_py = method_dir / "main.py"

    if not main_py.exists():
        print(f"[错误] {method_name}/main.py 不存在，跳过")
        return None

    # 创建结果目录
    method_result_dir = RESULTS_DIR / method_name
    method_result_dir.mkdir(parents=True, exist_ok=True)

    output_file = method_result_dir / "result.json"
    log_file = method_result_dir / "training.log"

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"开始运行: {method_name}")
    print(f"开始时间: {start_timestamp}")
    print(f"计划: 最多 {MAX_EPOCHS} epochs")
    print(f"时间限制: 45 分钟")
    print(f"{'='*60}")

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    # 构建命令 - 使用timeout限制总时间（45分钟）
    cmd = [
        'timeout', '2700',  # 45分钟总超时
        sys.executable, '-u',  # 无缓冲
        str(main_py),
        '--band', BAND,
        '--epochs', str(MAX_EPOCHS),
        '--batch-size', '8',  # 降低batch size避免OOM
        '--output', str(output_file),
    ]

    # 启动进程
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
            last_step = 0

            for line in process.stdout:
                log_f.write(line)
                log_f.flush()

                # 解析进度
                if 'Epoch' in line:
                    import re
                    m = re.search(r'Epoch\s*\[(\d+)/(\d+)\]', line)
                    if m:
                        last_epoch = int(m.group(1))
                    m = re.search(r'PSNR[:\s]+([\d.]+)', line, re.I)
                    if m:
                        last_psnr = float(m.group(1))
                    m = re.search(r'Step\s*(\d+)', line, re.I)
                    if m:
                        last_step = int(m.group(1))

            process.wait()
            returncode = process.returncode

            if returncode == 124:  # timeout退出码
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
            "band": BAND,
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
    print("="*60)
    print("FY4B Super Resolution - lv1_macro 批量运行 v2")
    print(f"每个方法: 最多 {MAX_EPOCHS} epochs")
    print(f"总时间限制: 45分钟/方法")
    print("="*60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, method in enumerate(METHODS, 1):
        print(f"\n[{i}/{len(METHODS)}] 准备运行 {method}...")

        result = run_method(method)

        if result:
            all_results.append({
                "method": method,
                "start_time": result.get("start_time"),
                "end_time": result.get("end_time"),
                "runtime_seconds": result.get("runtime_seconds"),
                "epochs": result.get("train_epochs"),
                "val_psnr": result.get("val_psnr"),
                "status": result.get("status"),
            })

        # 保存中间结果
        summary_file = RESULTS_DIR / "summary_v2.json"
        with open(summary_file, "w") as f:
            json.dump({
                "total_methods": len(METHODS),
                "completed": len(all_results),
                "results": all_results,
            }, f, indent=2)

        print(f"\n[休息] 5秒后继续...")
        time.sleep(5)

    # 最终汇总
    print("\n" + "="*60)
    print("所有方法运行完成!")
    print("="*60)
    print(f"\n{'方法':<25} {'时间':<10} {'Epochs':<8} {'PSNR':<10}")
    print("-"*55)
    for r in all_results:
        runtime = f"{r['runtime_seconds']/60:.1f}m" if r.get('runtime_seconds') else 'N/A'
        epochs = str(r.get('epochs', 'N/A'))
        psnr = f"{r['val_psnr']:.2f}" if r.get('val_psnr') else 'N/A'
        print(f"{r['method']:<25} {runtime:<10} {epochs:<8} {psnr:<10}")

    print(f"\n汇总文件: {RESULTS_DIR / 'summary_v2.json'}")

if __name__ == "__main__":
    main()
