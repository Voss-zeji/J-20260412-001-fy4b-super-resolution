#!/usr/bin/env python3
"""
批量运行 lv2_micro 下所有融合方法，做快速收敛性测试
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
MAX_EPOCHS = 10  # 快速测试：10 epochs
TIMEOUT_SECONDS = 900  # 15分钟总超时
BAND = "CH07"

METHODS = [
    "10_method_swinrestorer",
    "11_method_edgepft",
    "12_method_latentswin",
    "13_method_mambapft",
    "14_method_dualscalerestore",
]

FEISHU_WEBHOOK = os.environ.get("FEISHU_WEBHOOK_URL", "")


def send_feishu(action, method_name, extra=""):
    """发送飞书通知"""
    project_root = Path(__file__).parent.parent
    feishu_script = project_root / "send_feishu.py"
    if feishu_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(feishu_script), action, method_name, extra, ""],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass


def run_method(method_name: str):
    """运行单个方法"""
    method_dir = METHODS_DIR / method_name
    main_py = method_dir / "main.py"

    if not main_py.exists():
        print(f"[错误] {method_name}/main.py 不存在，跳过")
        return None

    method_result_dir = RESULTS_DIR / method_name
    method_result_dir.mkdir(parents=True, exist_ok=True)

    output_file = method_result_dir / "result.json"
    log_file = method_result_dir / "training.log"

    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"开始运行: {method_name}")
    print(f"开始时间: {start_timestamp}")
    print(f"计划: 最多 {MAX_EPOCHS} epochs")
    print(f"时间限制: {TIMEOUT_SECONDS//60} 分钟")
    print(f"{'='*60}")

    send_feishu("开始", method_name, f"lv2快速测试 {MAX_EPOCHS}epochs")

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    cmd = [
        'timeout', str(TIMEOUT_SECONDS),
        sys.executable, '-u',
        str(main_py),
        '--band', BAND,
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
                    m = re.search(r'val_psnr:\s*([\d.]+)', line, re.I)
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
        send_feishu("异常", method_name, str(e))
        return None

    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    actual_runtime = (end_time - start_time).total_seconds()

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

    send_feishu("完成", method_name, f"PSNR={result_data.get('val_psnr', 'N/A')} epochs={result_data.get('train_epochs', 'N/A')}")

    return result_data


def main():
    print("="*60)
    print("FY4B Super Resolution - lv2_micro 融合方法快速测试")
    print(f"每个方法: 最多 {MAX_EPOCHS} epochs")
    print(f"总时间限制: {TIMEOUT_SECONDS//60}分钟/方法")
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

        summary_file = RESULTS_DIR / "summary_lv2_quick.json"
        with open(summary_file, "w") as f:
            json.dump({
                "total_methods": len(METHODS),
                "completed": len(all_results),
                "results": all_results,
            }, f, indent=2)

        if i < len(METHODS):
            print(f"\n[休息] 5秒后继续...")
            time.sleep(5)

    print("\n" + "="*60)
    print("所有方法运行完成!")
    print("="*60)
    print(f"\n{'方法':<30} {'时间':<10} {'Epochs':<8} {'PSNR':<10}")
    print("-"*58)
    for r in all_results:
        runtime = f"{r['runtime_seconds']/60:.1f}m" if r.get('runtime_seconds') else 'N/A'
        epochs = str(r.get('epochs', 'N/A'))
        psnr = f"{r['val_psnr']:.2f}" if r.get('val_psnr') else 'N/A'
        print(f"{r['method']:<30} {runtime:<10} {epochs:<8} {psnr:<10}")

    print(f"\n汇总文件: {RESULTS_DIR / 'summary_lv2_quick.json'}")


if __name__ == "__main__":
    main()
