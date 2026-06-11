#!/usr/bin/env python3
"""
新论文方法批量训练脚本 - 第二批
训练 31_sfg_swinir, 33_dual_branch_emambair, 34_sfg_pftsr，200 epoch，CH07
在第一批(30, 32)完成后执行
"""

import subprocess, json, os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
METHODS_DIR = PROJECT_ROOT / 'lv2_micro' / 'methods_new'
PYTHON_BIN = '/root/miniconda3/envs/mamba2/bin/python'
RESULTS_BASE = PROJECT_ROOT / 'lv3_fusion' / 'results'

METHODS = {
    '31_method_sfg_swinir': METHODS_DIR / '31_method_sfg_swinir' / 'main.py',
    '33_method_dual_branch_emambair': METHODS_DIR / '33_method_dual_branch_emambair' / 'main.py',
    '34_method_sfg_pftsr': METHODS_DIR / '34_method_sfg_pftsr' / 'main.py',
}

def run_method(name, main_py):
    result_dir = RESULTS_BASE / name
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / 'result.json'
    log_file = result_dir / 'training.log'

    if output_file.exists():
        try:
            with open(output_file) as f:
                data = json.load(f)
            if data.get('train_epochs') == 200 and data.get('status') == 'success':
                print(f'  {name}: 已有 200ep 结果，跳过')
                return data
        except:
            pass

    cmd = [
        PYTHON_BIN, '-u', str(main_py),
        '--band', 'CH07',
        '--epochs', '200',
        '--batch-size', '8',
        '--lr', '0.0001',
        '--output', str(output_file),
    ]

    print(f'\n[{name}] 开始训练...')
    print(f'  命令: {" ".join(cmd)}')

    with open(log_file, 'w') as f:
        f.write(f'[{datetime.now()}] Start {name}\n')
        f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, universal_newlines=True)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(f'  {line.rstrip()}')
        proc.wait()

    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)
    return None

def main():
    print('=' * 60)
    print('新论文方法批量训练 - 第二批 (31, 33, 34)')
    print('=' * 60)

    all_results = []
    for name, main_py in METHODS.items():
        if not main_py.exists():
            print(f'[错误] {main_py} 不存在')
            continue
        result = run_method(name, main_py)
        if result:
            all_results.append(result)

    print('\n' + '=' * 60)
    print('训练完成!')
    print('=' * 60)
    for r in all_results:
        psnr = r.get('best_psnr') or r.get('final_psnr')
        print(f"  {r['method']}: PSNR={psnr:.2f} dB")

if __name__ == '__main__':
    main()
