#!/usr/bin/env python3
"""
LV3 统一深度训练脚本
8 个保留方法，200 epoch，CH07 通道
每 50 epoch 保存 checkpoint 和指标快照
"""

import subprocess
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
LV3_DIR = Path(__file__).parent
METHODS_DIR = LV3_DIR / "methods"
RESULTS_DIR = LV3_DIR / "results"
PYTHON_BIN = "/root/miniconda3/envs/mamba2/bin/python"

MAX_EPOCHS = 200
BATCH_SIZE = 8
BAND = "CH07"
CHECKPOINT_EVERY = 50  # 每 50 epoch 记录一次

# 8 个保留方法：4 个来自 LV1，4 个来自 LV2
# 源码位置映射
METHOD_SOURCES = {
    "04_method_pftsr": PROJECT_ROOT / "lv1_macro" / "methods" / "04_method_pftsr",
    "05_method_swinir": PROJECT_ROOT / "lv1_macro" / "methods" / "05_method_swinir",
    "06_method_tinynina": PROJECT_ROOT / "lv1_macro" / "methods" / "06_method_tinynina",
    "08_method_realrestorer": PROJECT_ROOT / "lv1_macro" / "methods" / "08_method_realrestorer",
    "10_method_swinrestorer": PROJECT_ROOT / "lv2_micro" / "lv2-save" / "10_method_swinrestorer",
    "11_method_edgepft": PROJECT_ROOT / "lv2_micro" / "lv2-save" / "11_method_edgepft",
    "14_method_dualscalerestore": PROJECT_ROOT / "lv2_micro" / "lv2-save" / "14_method_dualscalerestore",
    "17_method_emambair": PROJECT_ROOT / "lv2_micro" / "lv2-save" / "17_method_emambair",
}

METHODS = list(METHOD_SOURCES.keys())


def run_method(method_name: str):
    """运行单个方法 200 epoch"""
    source_dir = METHOD_SOURCES[method_name]
    main_py = source_dir / "main.py"

    if not main_py.exists():
        print(f"[错误] {main_py} 不存在，跳过")
        return None

    # 创建方法结果目录
    result_dir = RESULTS_DIR / method_name
    result_dir.mkdir(parents=True, exist_ok=True)

    output_file = result_dir / "result.json"
    log_file = result_dir / "training.log"
    snapshot_file = result_dir / "snapshots.json"

    start_time = datetime.now()
    start_ts = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"开始训练: {method_name}")
    print(f"来源: {source_dir.relative_to(PROJECT_ROOT)}")
    print(f"开始时间: {start_ts}")
    print(f"配置: {MAX_EPOCHS} epochs, batch={BATCH_SIZE}, band={BAND}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    cmd = [
        PYTHON_BIN, '-u',
        str(main_py),
        '--band', BAND,
        '--epochs', str(MAX_EPOCHS),
        '--batch-size', str(BATCH_SIZE),
        '--output', str(output_file),
    ]

    snapshots = []
    last_epoch = 0
    last_psnr = 0.0
    last_ssim = 0.0
    last_loss = 0.0

    try:
        with open(log_file, "w") as log_f:
            log_f.write(f"[{start_ts}] 开始训练 {method_name}\n")
            log_f.write(f"命令: {' '.join(cmd)}\n")
            log_f.write("=" * 60 + "\n")
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

                # 解析 epoch 行
                m = re.search(r'Epoch\s*\[(\d+)/\d+\]', line)
                if m:
                    last_epoch = int(m.group(1))

                m = re.search(r'[Vv]al\s*[Pp][Ss][Nn][Rr][:\s]+([\d.]+)', line)
                if m:
                    last_psnr = float(m.group(1))

                m = re.search(r'[Vv]al\s*[Ss][Ss][Ii][Mm][:\s]+([\d.]+)', line)
                if m:
                    last_ssim = float(m.group(1))

                m = re.search(r'[Ll]oss[:\s]+([\d.]+)', line)
                if m:
                    last_loss = float(m.group(1))

                # 每 CHECKPOINT_EVERY epoch 记录快照
                if last_epoch > 0 and last_epoch % CHECKPOINT_EVERY == 0:
                    # 避免同一 epoch 重复记录
                    if not snapshots or snapshots[-1]["epoch"] != last_epoch:
                        snap = {
                            "epoch": last_epoch,
                            "psnr": last_psnr,
                            "ssim": last_ssim,
                            "loss": last_loss,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        snapshots.append(snap)
                        print(f"  📌 Epoch {last_epoch}: PSNR={last_psnr:.2f}, SSIM={last_ssim:.4f}")

                        # 保存快照
                        with open(snapshot_file, "w") as sf:
                            json.dump(snapshots, sf, indent=2)

            process.wait()
            returncode = process.returncode

            if returncode != 0:
                log_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 进程退出码: {returncode}\n")
            else:
                log_f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 训练正常完成\n")

    except Exception as e:
        print(f"[错误] 训练 {method_name} 时出错: {e}")
        return None

    end_time = datetime.now()
    end_ts = end_time.strftime("%Y-%m-%d %H:%M:%S")
    runtime = (end_time - start_time).total_seconds()

    # 读取最终结果
    result_data = {}
    if output_file.exists():
        try:
            with open(output_file) as f:
                result_data = json.load(f)
        except Exception:
            pass

    if not result_data:
        result_data = {
            "method": method_name,
            "band": BAND,
            "status": "partial" if last_epoch < MAX_EPOCHS else "success",
            "epochs": last_epoch,
            "best_psnr": last_psnr if last_psnr > 0 else None,
        }

    result_data["runtime_seconds"] = runtime
    result_data["start_time"] = start_ts
    result_data["end_time"] = end_ts
    result_data["snapshots"] = snapshots

    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"结束时间: {end_ts}")
    print(f"运行时长: {runtime/3600:.1f} 小时")
    print(f"最终: Epoch {last_epoch}/{MAX_EPOCHS}, PSNR={last_psnr:.2f}, SSIM={last_ssim:.4f}")

    return result_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='LV3 统一深度训练')
    parser.add_argument('--methods', nargs='+', help='指定方法编号，如 04 05 06')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--band', type=str, default='CH07')
    args = parser.parse_args()

    global MAX_EPOCHS, BAND
    MAX_EPOCHS = args.epochs
    BAND = args.band

    # 过滤方法
    if args.methods:
        filtered = [m for m in METHODS if any(m.startswith(f"{n}_") for n in args.methods)]
    else:
        filtered = METHODS

    print("=" * 60)
    print("FY-4B SR — LV3 统一深度训练")
    print(f"方法数: {len(filtered)}")
    print(f"Epochs: {MAX_EPOCHS}")
    print(f"Band: {BAND}")
    print(f"方法: {', '.join(filtered)}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, method in enumerate(filtered, 1):
        print(f"\n[{i}/{len(filtered)}] {method}")
        result = run_method(method)
        if result:
            all_results.append(result)

    # 保存汇总
    summary = {
        "phase": "LV3_deep_training",
        "band": BAND,
        "max_epochs": MAX_EPOCHS,
        "total_methods": len(filtered),
        "completed": len(all_results),
        "results": sorted(all_results, key=lambda x: x.get("best_psnr") or x.get("val_psnr") or 0, reverse=True),
        "generated_at": datetime.now().isoformat(),
    }

    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印最终排名
    print(f"\n{'='*60}")
    print(f"LV3 深度训练完成!")
    print(f"{'='*60}")
    print(f"\n最终排名 (PSNR 降序):")
    for i, r in enumerate(summary["results"], 1):
        psnr = r.get("best_psnr") or r.get("val_psnr")
        psnr_str = f"{psnr:.2f}" if psnr else "N/A"
        ssim = r.get("best_ssim") or r.get("val_ssim")
        ssim_str = f"{ssim:.4f}" if ssim else "N/A"
        ep = r.get("epochs") or r.get("train_epochs", "?")
        rt = r.get("runtime_seconds", 0)
        print(f"  {i}. {r['method']}: PSNR={psnr_str}, SSIM={ssim_str}, "
              f"Epochs={ep}, Runtime={rt/3600:.1f}h")


if __name__ == "__main__":
    main()
