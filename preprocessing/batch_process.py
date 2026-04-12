#!/usr/bin/env python3
"""
FY-4B AGRI L1 数据批量定标处理脚本
支持多进程并行处理
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fy4b_calibration import FY4BCalibrator, save_calibrated_data


def process_file(args):
    """处理单个文件的包装函数（用于多进程）"""
    input_file, output_dir, method = args
    try:
        calibrator = FY4BCalibrator(input_file)
        output_path = save_calibrated_data(calibrator, output_dir, method=method)
        return (input_file, True, output_path, None)
    except Exception as e:
        return (input_file, False, None, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='FY-4B AGRI L1 数据批量定标处理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 处理单个文件
  python batch_process.py -i /data/FY4B_2000M/20250322.HDF -o ./output

  # 批量处理目录
  python batch_process.py -i /data/FY4B_2000M/ -o ./output --batch

  # 多进程批量处理 (使用4个进程)
  python batch_process.py -i /data/FY4B_2000M/ -o ./output --batch -p 4

  # 只处理特定波段时间的数据
  python batch_process.py -i /data/FY4B_2000M/ -o ./output --batch --pattern "*20250322*.HDF"
        '''
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入文件或目录路径')
    parser.add_argument('-o', '--output', type=str, default='./calibrated',
                        help='输出目录路径 (默认: ./calibrated)')
    parser.add_argument('-b', '--batch', action='store_true',
                        help='批量处理模式')
    parser.add_argument('-m', '--method', type=str, default='lut',
                        choices=['lut', 'linear'],
                        help='定标方法: lut(查找表) 或 linear(线性)')
    parser.add_argument('-p', '--processes', type=int, default=1,
                        help='并行进程数 (默认: 1, 建议不超过CPU核心数)')
    parser.add_argument('--pattern', type=str, default='*.HDF',
                        help='文件匹配模式 (默认: *.HDF)')
    parser.add_argument('--compress', action='store_true', default=True,
                        help='启用HDF5压缩 (默认: True)')
    parser.add_argument('--no-compress', dest='compress', action='store_false',
                        help='禁用HDF5压缩')
    parser.add_argument('--bands', type=str, default='all',
                        help='指定处理的波段，如 "01,02,07" 或 "all"')
    
    args = parser.parse_args()
    
    # 收集输入文件
    if args.batch:
        search_pattern = os.path.join(args.input, args.pattern)
        input_files = sorted(glob.glob(search_pattern))
    else:
        if os.path.isfile(args.input):
            input_files = [args.input]
        else:
            print(f"错误: 文件不存在 - {args.input}")
            sys.exit(1)
    
    if not input_files:
        print(f"错误: 未找到匹配的文件")
        sys.exit(1)
    
    print("="*70)
    print("FY-4B AGRI L1 数据批量定标处理")
    print("="*70)
    print(f"\n输入文件数量: {len(input_files)}")
    print(f"输出目录: {args.output}")
    print(f"定标方法: {args.method}")
    print(f"并行进程: {args.processes}")
    print(f"HDF5压缩: {args.compress}")
    print(f"处理波段: {args.bands}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 准备处理参数
    process_args = [(f, args.output, args.method) for f in input_files]
    
    # 开始处理
    start_time = time.time()
    results = []
    
    if args.processes > 1 and len(input_files) > 1:
        # 多进程处理
        print(f"\n使用 {args.processes} 个进程并行处理...")
        with Pool(processes=args.processes) as pool:
            results = pool.map(process_file, process_args)
    else:
        # 单进程处理
        print("\n单进程顺序处理...")
        for i, arg in enumerate(process_args, 1):
            print(f"\n[{i}/{len(input_files)}] 处理: {os.path.basename(arg[0])}")
            result = process_file(arg)
            results.append(result)
            
            # 显示进度
            if result[1]:
                print(f"  ✓ 成功: {os.path.basename(result[2])}")
            else:
                print(f"  ✗ 失败: {result[3]}")
    
    # 统计结果
    elapsed_time = time.time() - start_time
    success_count = sum(1 for r in results if r[1])
    failed_count = len(results) - success_count
    
    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"总文件数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"用时: {elapsed_time:.2f} 秒")
    print(f"平均: {elapsed_time/len(results):.2f} 秒/文件")
    
    # 显示失败的文件
    if failed_count > 0:
        print("\n失败的文件:")
        for r in results:
            if not r[1]:
                print(f"  - {os.path.basename(r[0])}: {r[3]}")
    
    # 生成处理报告
    report_path = os.path.join(args.output, 'processing_report.txt')
    with open(report_path, 'w') as f:
        f.write("FY-4B数据定标处理报告\n")
        f.write("="*70 + "\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"定标方法: {args.method}\n")
        f.write(f"处理波段: {args.bands}\n")
        f.write(f"\n处理统计:\n")
        f.write(f"  总文件数: {len(results)}\n")
        f.write(f"  成功: {success_count}\n")
        f.write(f"  失败: {failed_count}\n")
        f.write(f"  用时: {elapsed_time:.2f} 秒\n")
        f.write(f"\n文件列表:\n")
        for r in results:
            status = "成功" if r[1] else f"失败: {r[3]}"
            f.write(f"  {os.path.basename(r[0])} - {status}\n")
    
    print(f"\n处理报告已保存: {report_path}")


if __name__ == '__main__':
    main()
