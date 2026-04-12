#!/usr/bin/env python3
"""
FY-4B AGRI L1 数据批量定标处理 - 仅处理Channel07和Channel08
使用查找表定标方式

重要说明：
- 定标后的数据包含NaN值，这是正常的（对应原始数据中的填充值65535）
- 使用 np.nanmin()/np.nanmax() 获取有效数据的统计信息
- 使用 np.isnan() 检查NaN值分布
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
        
        # 保存CH07
        ch07_output = os.path.join(output_dir_ch07, basename.replace('_NOM_', '_CAL_'))
        with h5py.File(ch07_output, 'w') as f:
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
            dset.attrs['valid_pixels'] = stats['ch07_valid']
            dset.attrs['nan_pixels'] = stats['ch07_nan']
            dset.attrs['valid_ratio_%'] = ch07_valid_ratio
            dset.attrs['min'] = ch07_min
            dset.attrs['max'] = ch07_max
            dset.attrs['mean'] = ch07_mean
            dset.attrs['std'] = ch07_std
            
            # 复制原始文件元数据
            if hasattr(calibrator, 'file_attrs'):
                for key, val in calibrator.file_attrs.items():
                    try:
                        f.attrs[key] = val
                    except:
                        pass
        
        # 保存CH08
        ch08_output = os.path.join(output_dir_ch08, basename.replace('_NOM_', '_CAL_'))
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
            dset.attrs['note'] = 'NaN values represent fill values (invalid pixels)'
            
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
    
    # 获取所有HDF文件
    pattern = os.path.join(input_dir, '*.HDF')
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
            status = "✓" if result[1] else "✗"
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
    # 配置
    input_dirs = {
        '2000M': '/root/autodl-tmp/2000M',
        '4000M': '/root/autodl-tmp/4000M',
    }
    
    output_base = '/root/autodl-tmp/Calibration-FY4B'
    
    # 获取CPU核心数
    n_cores = cpu_count()
    n_processes = min(8, n_cores)  # 最多使用8个进程
    
    print("="*70)
    print("FY-4B AGRI L1 数据批量定标 - Channel07 & Channel08")
    print("="*70)
    print(f"CPU核心数: {n_cores}, 使用进程数: {n_processes}")
    print("\n重要提示:")
    print("  - 定标后的数据包含NaN值是正常的（对应原始数据中的填充值65535）")
    print("  - 使用 np.nanmin()/np.nanmax() 获取有效数据的统计信息")
    print("  - HDF文件中包含 valid_pixels/nan_pixels 属性用于验证")
    
    all_results = {}
    
    for res_name, input_dir in input_dirs.items():
        if not os.path.exists(input_dir):
            print(f"\n跳过不存在的目录: {input_dir}")
            continue
        
        output_dir_ch07 = os.path.join(output_base, res_name, 'CH07')
        output_dir_ch08 = os.path.join(output_base, res_name, 'CH08')
        
        results = batch_process_folder(input_dir, output_dir_ch07, output_dir_ch08, n_processes)
        all_results[res_name] = results
    
    # 最终统计
    print("\n" + "="*70)
    print("处理完成统计")
    print("="*70)
    
    total_files = 0
    total_success = 0
    
    for res_name, results in all_results.items():
        success = sum(1 for r in results if r[1])
        failed = len(results) - success
        total_files += len(results)
        total_success += success
        
        print(f"\n{res_name}:")
        print(f"  总文件: {len(results)}")
        print(f"  成功: {success}")
        print(f"  失败: {failed}")
        
        if failed > 0:
            print("  失败的文件:")
            for r in results:
                if not r[1]:
                    print(f"    - {os.path.basename(r[0])}: {r[4]}")
    
    print(f"\n总计: {total_success}/{total_files} 个文件成功处理")
    print("="*70)
    
    # 保存处理日志
    log_file = os.path.join(output_base, 'processing_log.txt')
    with open(log_file, 'w') as f:
        f.write("FY-4B Channel07 & Channel08 定标处理日志\n")
        f.write("="*70 + "\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"定标方法: LUT (查找表)\n")
        f.write(f"处理波段: Channel07 (IR3.90), Channel08 (IR7.00)\n")
        f.write("\n重要提示:\n")
        f.write("  - 定标后的数据包含NaN值是正常的（对应原始数据中的填充值65535）\n")
        f.write("  - 使用 np.nanmin()/np.nanmax() 获取有效数据的统计信息\n")
        f.write("  - 每个HDF文件包含 valid_pixels/nan_pixels 属性\n\n")
        
        for res_name, results in all_results.items():
            f.write(f"\n{res_name}:\n")
            for r in results:
                basename = os.path.basename(r[0])
                if r[1]:
                    stats = r[5]
                    f.write(f"  [OK] {basename}\n")
                    f.write(f"       CH07: {r[2]}\n")
                    f.write(f"             valid: {stats['ch07_valid']:,}, nan: {stats['ch07_nan']:,}\n")
                    f.write(f"       CH08: {r[3]}\n")
                    f.write(f"             valid: {stats['ch08_valid']:,}, nan: {stats['ch08_nan']:,}\n")
                else:
                    f.write(f"  [FAIL] {basename}: {r[4]}\n")
    
    print(f"\n处理日志已保存: {log_file}")
    
    # 数据读取示例
    print("\n" + "="*70)
    print("数据读取示例代码:")
    print("="*70)
    print("""
import h5py
import numpy as np

# 读取定标后的数据
with h5py.File('calibrated_file.HDF', 'r') as f:
    data = f['Channel07'][()]
    
    # 正确的统计方法（跳过NaN）
    valid_data = data[~np.isnan(data)]
    print(f"有效像素数: {len(valid_data)}")
    print(f"温度范围: {np.nanmin(data):.2f} K ~ {np.nanmax(data):.2f} K")
    print(f"平均温度: {np.nanmean(data):.2f} K")
    
    # 查看属性
    print(f"属性: {dict(f['Channel07'].attrs)}")
""")


if __name__ == '__main__':
    main()
