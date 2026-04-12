#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FY-4B新数据批量定标处理脚本
处理 /root/autodl-tmp/FY-4B 下的原始数据为定标后数据

使用方法:
    python batch_calibrate_new_data.py --input-base /root/autodl-tmp/FY-4B \
                                       --output-base /root/autodl-tmp/Calibration-FY4B-New \
                                       --channel CH07
"""

import os
import sys
import argparse
import glob
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from functools import partial

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.fy4b_calibration import FY4BCalibrator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FY-4B新数据批量定标')
    parser.add_argument('--input-base', type=str, default='/root/autodl-tmp/FY-4B',
                        help='输入数据根目录')
    parser.add_argument('--output-base', type=str, default='/root/autodl-tmp/Calibration-FY4B-New',
                        help='输出数据根目录')
    parser.add_argument('--channel', type=str, default='CH07', choices=['CH07', 'CH08'],
                        help='处理的通道')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行处理的工作进程数')
    return parser.parse_args()


def find_hdf_files(input_base, resolution):
    """
    查找指定分辨率的所有HDF文件
    
    文件路径格式: FY-4B/日期/FDI-/分辨率/*.HDF
    """
    pattern = os.path.join(input_base, '*', 'FDI-', resolution, '*.HDF')
    files = glob.glob(pattern)
    # 过滤掉临时文件
    files = [f for f in files if not f.endswith('.tmp')]
    return sorted(files)


def process_single_file(args):
    """
    处理单个HDF文件
    
    Args:
        args: (input_file, output_dir, channel_name) 元组
    """
    input_file, output_dir, channel_name = args
    
    try:
        # 创建定标器
        calibrator = FY4BCalibrator(input_file)
        
        # 读取DN值
        with h5py.File(input_file, 'r') as f:
            dn_data = f[f'Data/NOM{channel_name}'][()]
        
        # 使用查找表定标
        calibrated_data = calibrator.calibrate_with_lut(channel_name, dn_data)
        
        # 处理NaN值（使用邻近有效值或平均值填充）
        if np.any(~np.isfinite(calibrated_data)):
            nan_mask = ~np.isfinite(calibrated_data)
            valid_mean = np.nanmean(calibrated_data)
            calibrated_data = np.where(nan_mask, valid_mean, calibrated_data)
        
        # 生成输出文件名（添加CAL标记）
        basename = os.path.basename(input_file)
        # 将 NOM 替换为 CAL
        output_basename = basename.replace('NOM_', 'CAL_')
        output_file = os.path.join(output_dir, output_basename)
        
        # 保存定标后的数据
        with h5py.File(output_file, 'w') as f:
            # 创建数据集
            ds = f.create_dataset(channel_name, data=calibrated_data, compression='gzip')
            
            # 添加属性
            ds.attrs['units'] = 'K' if channel_name in ['Channel07', 'Channel08'] else 'reflectance'
            ds.attrs['description'] = f'FY-4B {channel_name} calibrated data'
            ds.attrs['source_file'] = input_file
            ds.attrs['calibration_time'] = datetime.now().isoformat()
            
            # 复制原始文件的地理定位信息（如果有）
            with h5py.File(input_file, 'r') as src:
                for attr_name, attr_value in src.attrs.items():
                    try:
                        f.attrs[attr_name] = attr_value
                    except:
                        pass
        
        return {'status': 'success', 'input': input_file, 'output': output_file}
        
    except Exception as e:
        return {'status': 'error', 'input': input_file, 'error': str(e)}


def main():
    """主函数"""
    args = parse_args()
    
    # 通道名称转换
    channel_name = args.channel.replace('CH', 'Channel')
    
    print("="*60)
    print("FY-4B 新数据批量定标处理")
    print("="*60)
    print(f"输入目录: {args.input_base}")
    print(f"输出目录: {args.output_base}")
    print(f"处理通道: {args.channel} ({channel_name})")
    print(f"工作进程: {args.workers}")
    print("="*60)
    
    # 处理2000M和4000M数据
    resolutions = ['2000M', '4000M']
    
    for resolution in resolutions:
        print(f"\n处理 {resolution} 数据...")
        
        # 查找输入文件
        input_files = find_hdf_files(args.input_base, resolution)
        print(f"  找到 {len(input_files)} 个输入文件")
        
        if len(input_files) == 0:
            print(f"  跳过 {resolution}")
            continue
        
        # 创建输出目录
        output_dir = os.path.join(args.output_base, resolution, args.channel)
        os.makedirs(output_dir, exist_ok=True)
        print(f"  输出目录: {output_dir}")
        
        # 准备任务列表
        tasks = [(f, output_dir, channel_name) for f in input_files]
        
        # 并行处理
        print(f"  开始定标处理 (使用 {args.workers} 个进程)...")
        
        success_count = 0
        error_count = 0
        
        with mp.Pool(processes=args.workers) as pool:
            with tqdm(total=len(tasks), desc=f"  {resolution}") as pbar:
                for result in pool.imap_unordered(process_single_file, tasks):
                    if result['status'] == 'success':
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"\n    错误: {result['input']} - {result['error']}")
                    pbar.update(1)
        
        print(f"  完成: {success_count} 成功, {error_count} 失败")
    
    print("\n" + "="*60)
    print("所有处理完成!")
    print("="*60)


if __name__ == '__main__':
    main()
