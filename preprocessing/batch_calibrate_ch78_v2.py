#!/usr/bin/env python3
"""
FY-4B AGRI L1 数据批量定标处理 - 支持 NOM 和 CAL 两种格式
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


def detect_file_format(filename):
    """检测文件是 NOM 还是 CAL 格式"""
    basename = os.path.basename(filename)
    if '_NOM_' in basename:
        return 'NOM'
    elif '_CAL_' in basename:
        return 'CAL'
    else:
        return 'UNKNOWN'


def process_single_file(args):
    """
    处理单个文件，提取并定标CH07和CH08

    Args:
        args: (input_file, output_dir_ch07, output_dir_ch08)

    Returns:
        (input_file, success, ch07_path, ch08_path, error_msg, stats)
    """
    input_file, output_dir_ch07, output_dir_ch08 = args
    basename = os.path.basename(input_file)
    file_format = detect_file_format(input_file)
    stats = {'ch07_valid': 0, 'ch07_nan': 0, 'ch08_valid': 0, 'ch08_nan': 0}

    try:
        # 创建定标器
        calibrator = FY4BCalibrator(input_file)

        # 定标CH07和CH08
        ch07_data = calibrator.calibrate_with_lut('Channel07')
        ch08_data = calibrator.calibrate_with_lut('Channel08')

        # 数据验证和统计
        stats['ch07_valid'] = int(np.sum(~np.isnan(ch07_data)))
        stats['ch07_nan'] = int(np.sum(np.isnan(ch07_data)))
        stats['ch08_valid'] = int(np.sum(~np.isnan(ch08_data)))
        stats['ch08_nan'] = int(np.sum(np.isnan(ch08_data)))

        total_pixels = ch07_data.size
        ch07_valid_ratio = stats['ch07_valid'] / total_pixels * 100
        ch08_valid_ratio = stats['ch08_valid'] / total_pixels * 100

        # 检查是否全为NaN（严重错误）
        if stats['ch07_valid'] == 0:
            raise ValueError(f"CH07 定标后全为NaN! 原始数据可能有问题。")
        if stats['ch08_valid'] == 0:
            raise ValueError(f"CH08 定标后全为NaN! 原始数据可能有问题。")

        # 计算统计值（仅针对有效数据）
        ch07_min = float(np.nanmin(ch07_data))
        ch07_max = float(np.nanmax(ch07_data))
        ch07_mean = float(np.nanmean(ch07_data))
        ch07_std = float(np.nanstd(ch07_data))

        ch08_min = float(np.nanmin(ch08_data))
        ch08_max = float(np.nanmax(ch08_data))
        ch08_mean = float(np.nanmean(ch08_data))
        ch08_std = float(np.nanstd(ch08_data))

        # 生成输出文件名
        # 例如: FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250301000000... -> FY4B_CH07_CAL_20250301000000.HDF
        timestamp = None
        parts = basename.split('_')
        for part in parts:
            if len(part) == 14 and part.isdigit():
                timestamp = part
                break

        if timestamp is None:
            raise ValueError(f"无法从文件名提取时间戳: {basename}")

        ch07_output = os.path.join(output_dir_ch07, f'FY4B_CH07_CAL_{timestamp}.HDF')
        ch08_output = os.path.join(output_dir_ch08, f'FY4B_CH08_CAL_{timestamp}.HDF')

        # 保存CH07
        with h5py.File(ch07_output, 'w') as f:
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
            dset.attrs['valid_pixels'] = stats['ch07_valid']
            dset.attrs['nan_pixels'] = stats['ch07_nan']
            dset.attrs['valid_ratio_%'] = ch07_valid_ratio
            dset.attrs['min'] = ch07_min
            dset.attrs['max'] = ch07_max
            dset.attrs['mean'] = ch07_mean
            dset.attrs['std'] = ch07_std

            if hasattr(calibrator, 'file_attrs'):
                for key, val in calibrator.file_attrs.items():
                    try:
                        f.attrs[key] = val
                    except:
                        pass

        # 保存CH08
        with h5py.File(ch08_output, 'w') as f:
            dset = f.create_dataset(
                'Channel08',
                data=ch08_data,
                compression='gzip',
                compression_opts=4,
                chunks=True
            )
            dset.attrs['band_name'] = 'IR7.00'
            dset.attrs['wavelength'] = '7.00μm'
            dset.attrs['type'] = 'brightness_temperature'
            dset.attrs['unit'] = 'K'
            dset.attrs['calibration_method'] = 'LUT'
            dset.attrs['fill_value'] = np.nan
            dset.attrs['valid_pixels'] = stats['ch08_valid']
            dset.attrs['nan_pixels'] = stats['ch08_nan']
            dset.attrs['valid_ratio_%'] = ch08_valid_ratio
            dset.attrs['min'] = ch08_min
            dset.attrs['max'] = ch08_max
            dset.attrs['mean'] = ch08_mean
            dset.attrs['std'] = ch08_std

            if hasattr(calibrator, 'file_attrs'):
                for key, val in calibrator.file_attrs.items():
                    try:
                        f.attrs[key] = val
                    except:
                        pass

        return (input_file, True, ch07_output, ch08_output, None, stats)

    except Exception as e:
        return (input_file, False, None, None, str(e), stats)


def batch_process_folder(input_dir, output_dir_ch07, output_dir_ch08, n_processes=4):
    """批量处理一个文件夹中的所有HDF文件"""

    # 获取所有HDF文件（NOM 和 CAL 格式）
    pattern = os.path.join(input_dir, '*_NOM_*.HDF')
    files = sorted(glob.glob(pattern))

    if not files:
        pattern = os.path.join(input_dir, '*_CAL_*.HDF')
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"警告: 在 {input_dir} 中未找到HDF文件")
        return []

    print(f"\n处理目录: {input_dir}")
    print(f"找到 {len(files)} 个文件")
    print(f"输出目录: {output_dir_ch07} 和 {output_dir_ch08}")
    print(f"使用 {n_processes} 个进程并行处理")

    # 确保输出目录存在
    os.makedirs(output_dir_ch07, exist_ok=True)
    os.makedirs(output_dir_ch08, exist_ok=True)

    # 准备参数
    args_list = [(f, output_dir_ch07, output_dir_ch08) for f in files]

    # 并行处理
    start_time = time.time()
    results = []

    with Pool(processes=n_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_file, args_list), 1):
            results.append(result)
            status = "OK" if result[1] else "FAIL"
            basename = os.path.basename(result[0])
            stats = result[5]

            if result[1]:
                ch07_ratio = stats['ch07_valid'] / (stats['ch07_valid'] + stats['ch07_nan']) * 100
                ch08_ratio = stats['ch08_valid'] / (stats['ch08_valid'] + stats['ch08_nan']) * 100
                print(f"  [{i}/{len(files)}] {status} {basename}")
                print(f"      CH07: {stats['ch07_valid']:,} valid ({ch07_ratio:.1f}%), CH08: {stats['ch08_valid']:,} valid ({ch08_ratio:.1f}%)")
            else:
                print(f"  [{i}/{len(files)}] {status} {basename}")
                print(f"      错误: {result[4]}")

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r[1])

    print(f"\n完成: {success_count}/{len(files)} 个文件成功处理")
    print(f"用时: {elapsed:.1f} 秒, 平均: {elapsed/len(files):.1f} 秒/文件")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='FY-4B 批量定标处理 CH07/CH08')
    parser.add_argument('--date', type=str, default='20250301',
                        help='日期前缀，如 20250301,20250302 或 all')
    parser.add_argument('--res', type=str, default='both',
                        choices=['2000M', '4000M', 'both'], help='分辨率')
    parser.add_argument('--workers', type=int, default=8, help='并行进程数')
    args = parser.parse_args()

    raw_data_base = '/root/autodl-tmp/FY-4B/Raw-data'
    output_base = '/root/autodl-tmp/FY-4B/calibration'

    n_cores = cpu_count()
    n_processes = min(args.workers, n_cores)

    print("=" * 70)
    print("FY-4B AGRI L1 数据批量定标 - Channel07 & Channel08 (v2)")
    print("=" * 70)
    print(f"日期: {args.date}")
    print(f"分辨率: {args.res}")
    print(f"CPU核心数: {n_cores}, 使用进程数: {n_processes}")

    all_results = {}

    # 解析日期
    if args.date == 'all':
        dates = [f'202503{str(i).zfill(2)}' for i in range(1, 17)]
    else:
        dates = args.date.split(',')

    for date in dates:
        # 2000M
        if args.res in ['2000M', 'both']:
            input_dir = f'{raw_data_base}/{date}-DISK-2000M'
            if os.path.exists(input_dir):
                output_dir_ch07 = f'{output_base}/2000M/CH07'
                output_dir_ch08 = f'{output_base}/2000M/CH08'
                results = batch_process_folder(input_dir, output_dir_ch07, output_dir_ch08, n_processes)
                all_results[f'{date}-2000M'] = results
            else:
                print(f"\n跳过不存在的目录: {input_dir}")

        # 4000M
        if args.res in ['4000M', 'both']:
            input_dir = f'{raw_data_base}/{date}-DISK-4000M'
            if os.path.exists(input_dir):
                output_dir_ch07 = f'{output_base}/4000M/CH07'
                output_dir_ch08 = f'{output_base}/4000M/CH08'
                results = batch_process_folder(input_dir, output_dir_ch07, output_dir_ch08, n_processes)
                all_results[f'{date}-4000M'] = results
            else:
                print(f"\n跳过不存在的目录: {input_dir}")

    # 最终统计
    print("\n" + "=" * 70)
    print("处理完成统计")
    print("=" * 70)

    total_files = 0
    total_success = 0

    for key, results in all_results.items():
        if results:
            success = sum(1 for r in results if r[1])
            failed = len(results) - success
            total_files += len(results)
            total_success += success

            print(f"\n{key}:")
            print(f"  总文件: {len(results)}")
            print(f"  成功: {success}")
            print(f"  失败: {failed}")

            if failed > 0:
                print("  失败的文件:")
                for r in results:
                    if not r[1]:
                        print(f"    - {os.path.basename(r[0])}: {r[4]}")

    print(f"\n总计: {total_success}/{total_files} 个文件成功处理")


if __name__ == '__main__':
    main()