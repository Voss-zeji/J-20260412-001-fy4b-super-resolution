#!/usr/bin/env python3
"""
FY-4B AGRI L1 数据批量定标处理 - 仅处理Channel07 (CH07)
使用查找表定标方式
"""

import os
import sys
import glob
import time
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fy4b_calibration import FY4BCalibrator


def extract_timestamp(filename):
    """从文件名提取时间戳用于匹配"""
    # FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250322134500_20250322135959_2000M_V0001.HDF
    parts = filename.split('_')
    for part in parts:
        if len(part) == 14 and part.isdigit():  # 20250322134500
            return part
    return None


def process_single_file(args):
    """
    处理单个文件，提取并定标CH07
    
    Args:
        args: (input_file, output_dir)
    
    Returns:
        (input_file, success, output_path, error_msg, stats)
    """
    input_file, output_dir = args
    basename = os.path.basename(input_file)
    stats = {'valid': 0, 'nan': 0, 'min': 0, 'max': 0, 'mean': 0}
    
    try:
        # 创建定标器
        calibrator = FY4BCalibrator(input_file)
        
        # 定标CH07
        ch07_data = calibrator.calibrate_with_lut('Channel07')
        
        # 数据验证和统计
        stats['valid'] = int(np.sum(~np.isnan(ch07_data)))
        stats['nan'] = int(np.sum(np.isnan(ch07_data)))
        
        total_pixels = ch07_data.size
        valid_ratio = stats['valid'] / total_pixels * 100 if total_pixels > 0 else 0
        
        # 检查是否全为NaN（严重错误）
        if stats['valid'] == 0:
            raise ValueError(f"CH07 定标后全为NaN! 原始数据可能有问题。")
        
        # 计算统计值（仅针对有效数据）
        stats['min'] = float(np.nanmin(ch07_data))
        stats['max'] = float(np.nanmax(ch07_data))
        stats['mean'] = float(np.nanmean(ch07_data))
        stats['std'] = float(np.nanstd(ch07_data))
        
        # 生成输出文件名
        timestamp = extract_timestamp(basename)
        if timestamp:
            output_name = f"FY4B_CH07_CAL_{timestamp}.HDF"
        else:
            output_name = basename.replace('_NOM_', '_CAL_')
        
        output_path = os.path.join(output_dir, output_name)
        
        # 保存CH07
        with h5py.File(output_path, 'w') as f:
            # 创建数据集（使用压缩）
            dset = f.create_dataset(
                'Channel07', 
                data=ch07_data,
                compression='gzip',
                compression_opts=4,
                chunks=True
            )
            dset.attrs['band_name'] = 'IR3.90'
            dset.attrs['wavelength'] = '3.90μm'
            dset.attrs['type'] = 'brightness_temperature'
            dset.attrs['unit'] = 'K'
            dset.attrs['calibration_method'] = 'LUT'
            dset.attrs['fill_value'] = np.nan
            dset.attrs['note'] = 'NaN values represent fill values (invalid pixels)'
            
            # 添加统计信息
            dset.attrs['valid_pixels'] = stats['valid']
            dset.attrs['nan_pixels'] = stats['nan']
            dset.attrs['valid_ratio_%'] = valid_ratio
            dset.attrs['min'] = stats['min']
            dset.attrs['max'] = stats['max']
            dset.attrs['mean'] = stats['mean']
            dset.attrs['std'] = stats['std']
            dset.attrs['source_file'] = basename
            
            # 复制原始文件元数据
            if hasattr(calibrator, 'file_attrs'):
                for key, val in calibrator.file_attrs.items():
                    try:
                        f.attrs[key] = val
                    except:
                        pass
        
        return (input_file, True, output_path, None, stats)
        
    except Exception as e:
        return (input_file, False, None, str(e), stats)


def find_hdf_files(base_dir):
    """递归查找所有HDF文件，按分辨率分类"""
    files_2000m = []
    files_4000m = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.HDF') or file.endswith('.hdf'):
                full_path = os.path.join(root, file)
                if '2000M' in file or '2000M' in root:
                    files_2000m.append(full_path)
                elif '4000M' in file or '4000M' in root:
                    files_4000m.append(full_path)
    
    return sorted(files_2000m), sorted(files_4000m)


def batch_process_resolution(files, output_dir, resolution_name, n_processes=4):
    """批量处理特定分辨率的文件"""
    
    if not files:
        print(f"警告: 未找到 {resolution_name} 的HDF文件")
        return []
    
    print(f"\n处理 {resolution_name} 数据:")
    print(f"找到 {len(files)} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"使用 {n_processes} 个进程并行处理")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数
    args_list = [(f, output_dir) for f in files]
    
    # 并行处理
    start_time = time.time()
    results = []
    
    with Pool(processes=n_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_file, args_list), 1):
            results.append(result)
            status = "✓" if result[1] else "✗"
            basename = os.path.basename(result[0])
            stats = result[4]
            
            if result[1]:
                valid_ratio = stats['valid'] / (stats['valid'] + stats['nan']) * 100 if (stats['valid'] + stats['nan']) > 0 else 0
                print(f"  [{i}/{len(files)}] {status} {basename}")
                print(f"      CH07: {stats['valid']:,} valid ({valid_ratio:.1f}%), range: {stats['min']:.1f}~{stats['max']:.1f} K")
            else:
                print(f"  [{i}/{len(files)}] {status} {basename}")
                print(f"      错误: {result[3]}")
    
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r[1])
    
    print(f"\n{resolution_name} 完成: {success_count}/{len(files)} 个文件成功处理")
    print(f"用时: {elapsed:.1f} 秒, 平均: {elapsed/len(files):.1f} 秒/文件")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FY-4B CH07 批量定标工具')
    parser.add_argument('input_dir', type=str, help='输入目录 (包含日期子目录)')
    parser.add_argument('output_base', type=str, help='输出基础目录')
    parser.add_argument('--processes', '-p', type=int, default=None, help='并行进程数 (默认: CPU核心数)')
    
    args = parser.parse_args()
    
    # 设置进程数
    if args.processes is None:
        n_cores = cpu_count()
        n_processes = min(8, n_cores)
    else:
        n_processes = args.processes
    
    print("="*70)
    print("FY-4B AGRI L1 数据批量定标 - Channel07 (IR3.90)")
    print("="*70)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_base}")
    print(f"CPU核心数: {cpu_count()}, 使用进程数: {n_processes}")
    
    # 查找所有HDF文件
    print("\n扫描输入目录...")
    files_2000m, files_4000m = find_hdf_files(args.input_dir)
    print(f"找到 2000M: {len(files_2000m)} 个文件")
    print(f"找到 4000M: {len(files_4000m)} 个文件")
    
    # 处理2000M
    output_2000m = os.path.join(args.output_base, '2000M', 'CH07')
    results_2000m = batch_process_resolution(files_2000m, output_2000m, '2000M', n_processes)
    
    # 处理4000M
    output_4000m = os.path.join(args.output_base, '4000M', 'CH07')
    results_4000m = batch_process_resolution(files_4000m, output_4000m, '4000M', n_processes)
    
    # 最终统计
    print("\n" + "="*70)
    print("处理完成统计")
    print("="*70)
    
    total_files = len(files_2000m) + len(files_4000m)
    total_success = sum(1 for r in results_2000m if r[1]) + sum(1 for r in results_4000m if r[1])
    
    print(f"\n2000M:")
    print(f"  总文件: {len(files_2000m)}")
    print(f"  成功: {sum(1 for r in results_2000m if r[1])}")
    print(f"  失败: {len(results_2000m) - sum(1 for r in results_2000m if r[1])}")
    
    print(f"\n4000M:")
    print(f"  总文件: {len(files_4000m)}")
    print(f"  成功: {sum(1 for r in results_4000m if r[1])}")
    print(f"  失败: {len(results_4000m) - sum(1 for r in results_4000m if r[1])}")
    
    print(f"\n总计: {total_success}/{total_files} 个文件成功处理")
    print("="*70)
    
    # 保存处理日志
    log_file = os.path.join(args.output_base, 'calibration_log_ch07.txt')
    with open(log_file, 'w') as f:
        f.write("FY-4B Channel07 定标处理日志\n")
        f.write("="*70 + "\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录: {args.input_dir}\n")
        f.write(f"输出目录: {args.output_base}\n")
        f.write(f"定标方法: LUT (查找表)\n")
        f.write(f"处理波段: Channel07 (IR3.90)\n\n")
        
        f.write("2000M 处理结果:\n")
        for r in results_2000m:
            basename = os.path.basename(r[0])
            if r[1]:
                stats = r[4]
                f.write(f"  [OK] {basename} -> {os.path.basename(r[2])}\n")
                f.write(f"       valid: {stats['valid']:,}, nan: {stats['nan']:,}, range: {stats['min']:.2f}~{stats['max']:.2f} K\n")
            else:
                f.write(f"  [FAIL] {basename}: {r[3]}\n")
        
        f.write("\n4000M 处理结果:\n")
        for r in results_4000m:
            basename = os.path.basename(r[0])
            if r[1]:
                stats = r[4]
                f.write(f"  [OK] {basename} -> {os.path.basename(r[2])}\n")
                f.write(f"       valid: {stats['valid']:,}, nan: {stats['nan']:,}, range: {stats['min']:.2f}~{stats['max']:.2f} K\n")
            else:
                f.write(f"  [FAIL] {basename}: {r[3]}\n")
    
    print(f"\n处理日志已保存: {log_file}")
    
    # 生成数据对列表
    print("\n生成数据对列表...")
    generate_pair_list(args.output_base, results_2000m, results_4000m)
    
    return total_success, total_files


def generate_pair_list(output_base, results_2000m, results_4000m):
    """生成2000M和4000M的数据对列表"""
    
    # 提取时间戳
    def get_timestamp(result):
        if result[1]:  # 成功
            return extract_timestamp(os.path.basename(result[2]))
        return None
    
    timestamps_2000m = {get_timestamp(r): r[2] for r in results_2000m if r[1]}
    timestamps_4000m = {get_timestamp(r): r[2] for r in results_4000m if r[1]}
    
    # 找匹配的对
    common_timestamps = set(timestamps_2000m.keys()) & set(timestamps_4000m.keys())
    
    pair_list_path = os.path.join(output_base, 'data_pairs_ch07.txt')
    with open(pair_list_path, 'w') as f:
        f.write("# FY-4B CH07 数据对列表 (2000M -> 4000M)\n")
        f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 2000M文件数: {len(timestamps_2000m)}\n")
        f.write(f"# 4000M文件数: {len(timestamps_4000m)}\n")
        f.write(f"# 匹配对数: {len(common_timestamps)}\n\n")
        f.write("# 2000M_path,4000M_path,timestamp\n")
        
        for ts in sorted(common_timestamps):
            f.write(f"{timestamps_2000m[ts]},{timestamps_4000m[ts]},{ts}\n")
    
    print(f"数据对列表已保存: {pair_list_path}")
    print(f"  - 2000M文件数: {len(timestamps_2000m)}")
    print(f"  - 4000M文件数: {len(timestamps_4000m)}")
    print(f"  - 匹配对数: {len(common_timestamps)}")
    
    # 保存统计信息
    stats_path = os.path.join(output_base, 'calibration_stats_ch07.json')
    import json
    stats = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_dir': os.path.dirname(list(timestamps_2000m.values())[0]) if timestamps_2000m else '',
        'output_base': output_base,
        '2000M_total': len(timestamps_2000m),
        '4000M_total': len(timestamps_4000m),
        'pairs': len(common_timestamps),
        'pair_list': pair_list_path
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return len(common_timestamps)


if __name__ == '__main__':
    main()
