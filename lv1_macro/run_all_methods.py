#!/usr/bin/env python3
"""
依次运行 lv1_macro 下所有方法，每个方法运行30分钟
记录：开始时间、运行批次、截至时的精度
"""

import subprocess
import json
import time
import signal
import sys
import os
from datetime import datetime
from pathlib import Path

# 配置
METHODS_DIR = Path(__file__).parent / "methods"
RESULTS_DIR = Path(__file__).parent / "results"
RUNTIME_SECONDS = 30 * 60  # 30分钟
BAND = "CH07"  # 默认通道

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

def run_method_with_timeout(method_name: str, timeout: int = RUNTIME_SECONDS):
    """运行单个方法，带超时控制"""
    method_dir = METHODS_DIR / method_name
    main_py = method_dir / "main.py"

    if not main_py.exists():
        print(f"[错误] {method_name}/main.py 不存在，跳过")
        return None

    # 创建结果目录
    method_result_dir = RESULTS_DIR / method_name
    method_result_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件
    output_file = method_result_dir / "result.json"
    log_file = method_result_dir / "training.log"

    # 记录开始时间
    start_time = datetime.now()
    start_timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"开始运行: {method_name}")
    print(f"开始时间: {start_timestamp}")
    print(f"计划运行: {timeout/60} 分钟")
    print(f"{'='*60}")

    # 构建命令
    cmd = [
        sys.executable,
        str(main_py),
        "--band", BAND,
        "--epochs", "10000",  # 设置很大的epoch数，让timeout来控制
        "--output", str(output_file),
    ]

    # 启动进程
    process = None
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
            )

            # 实时读取输出并写入日志
            start_ts = time.time()
            last_batch = 0
            last_psnr = 0.0

            for line in process.stdout:
                elapsed = time.time() - start_ts
                remaining = timeout - elapsed

                # 写入日志
                log_f.write(line)
                log_f.flush()

                # 解析关键信息
                if "Epoch" in line or "epoch" in line.lower():
                    # 尝试提取epoch信息
                    try:
                        # 常见的格式: "Epoch [5/100]" 或 "Epoch: 5/100"
                        import re
                        match = re.search(r'[Ee]poch\s*\[?(\d+)', line)
                        if match:
                            last_batch = int(match.group(1))
                    except:
                        pass

                if "PSNR" in line or "psnr" in line.lower():
                    # 尝试提取PSNR值
                    try:
                        import re
                        match = re.search(r'PSNR[:\s]+([\d.]+)', line, re.IGNORECASE)
                        if match:
                            last_psnr = float(match.group(1))
                    except:
                        pass

                # 检查是否超时
                if elapsed >= timeout:
                    print(f"\n[超时] {method_name} 已运行 {timeout/60} 分钟，准备停止...")
                    log_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 达到时间限制，停止训练\n")
                    break

            # 终止进程
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    except Exception as e:
        print(f"[错误] 运行 {method_name} 时出错: {e}")
        if process and process.poll() is None:
            process.kill()
        return None

    # 记录结束时间
    end_time = datetime.now()
    end_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
    actual_runtime = (end_time - start_time).total_seconds()

    # 尝试读取结果文件
    result_data = {}
    if output_file.exists():
        try:
            with open(output_file) as f:
                result_data = json.load(f)
        except:
            pass

    # 如果结果文件中没有数据，使用解析的数据
    if not result_data:
        result_data = {
            "method": method_name,
            "band": BAND,
            "status": "timeout",
            "train_epochs": last_batch,
            "val_psnr": last_psnr,
        }
    else:
        result_data["status"] = result_data.get("status", "timeout")
        result_data["train_epochs"] = result_data.get("train_epochs", last_batch)
        if last_psnr > 0 and "val_psnr" not in result_data:
            result_data["val_psnr"] = last_psnr

    # 添加运行元信息
    result_data["runtime_seconds"] = actual_runtime
    result_data["start_time"] = start_timestamp
    result_data["end_time"] = end_timestamp

    # 保存更新后的结果
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"结束时间: {end_timestamp}")
    print(f"实际运行: {actual_runtime/60:.1f} 分钟")
    print(f"运行批次: {result_data.get('train_epochs', 'N/A')}")
    print(f"截至精度: {result_data.get('val_psnr', 'N/A')}")
    print(f"结果保存: {output_file}")

    return result_data

def main():
    """主函数：依次运行所有方法"""
    print("="*60)
    print("FY4B Super Resolution - lv1_macro 批量运行")
    print(f"每个方法运行时间: {RUNTIME_SECONDS/60} 分钟")
    print(f"通道: {BAND}")
    print("="*60)

    # 确保结果目录存在
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 汇总结果
    all_results = []

    for i, method in enumerate(METHODS, 1):
        print(f"\n[{i}/{len(METHODS)}] 准备运行 {method}...")

        result = run_method_with_timeout(method, RUNTIME_SECONDS)

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

        # 保存中间汇总结果
        summary_file = RESULTS_DIR / "summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "total_methods": len(METHODS),
                "completed": len(all_results),
                "runtime_per_method": RUNTIME_SECONDS,
                "results": all_results,
            }, f, indent=2)

        # 方法之间短暂休息，让GPU冷却
        if i < len(METHODS):
            print(f"\n[休息] 等待 10 秒后开始下一个方法...")
            time.sleep(10)

    # 打印最终汇总
    print("\n" + "="*60)
    print("所有方法运行完成!")
    print("="*60)
    print(f"\n{'方法':<25} {'开始时间':<20} {'批次':<8} {'精度(PSNR)':<12}")
    print("-"*65)
    for r in all_results:
        print(f"{r['method']:<25} {r['start_time']:<20} {str(r.get('epochs', 'N/A')):<8} {str(r.get('val_psnr', 'N/A')):<12}")

    print(f"\n详细结果保存在: {RESULTS_DIR}")
    print(f"汇总文件: {RESULTS_DIR / 'summary.json'}")

if __name__ == "__main__":
    main()
