"""
FY-4B AGRI L1 数据定标处理模块
将DN值转换为实际物理量（反射率/亮温）

FY-4B数据采用查找表(Lookup Table)定标方式:
- 可见光/近红外波段 (Channel01-06): 转换为反射率 (0-1.1431)
- 红外波段 (Channel07-08): 转换为亮温 (190-405K)

定标方式:
1. 查找表定标（推荐）: 物理量 = CALTable[DN]
2. 线性定标: 物理量 = DN × Scale + Offset
"""

import h5py
import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import glob


class FY4BCalibrator:
    """FY-4B AGRI L1数据定标器"""
    
    # 波段信息
    BAND_INFO = {
        'Channel01': {'name': 'VIS0.47', 'type': 'reflectance', 'wavelength': '0.47μm'},
        'Channel02': {'name': 'VIS0.65', 'type': 'reflectance', 'wavelength': '0.65μm'},
        'Channel03': {'name': 'VIS0.83', 'type': 'reflectance', 'wavelength': '0.83μm'},
        'Channel04': {'name': 'NIR1.38', 'type': 'reflectance', 'wavelength': '1.38μm'},
        'Channel05': {'name': 'NIR1.61', 'type': 'reflectance', 'wavelength': '1.61μm'},
        'Channel06': {'name': 'NIR2.25', 'type': 'reflectance', 'wavelength': '2.25μm'},
        'Channel07': {'name': 'IR3.90',  'type': 'brightness_temp', 'wavelength': '3.90μm'},
        'Channel08': {'name': 'IR7.00',  'type': 'brightness_temp', 'wavelength': '7.00μm'},
    }
    
    # 填充值
    FILL_VALUE_DN = 65535
    FILL_VALUE_PHYS = np.nan
    
    def __init__(self, hdf_file: str):
        """
        初始化定标器
        
        Args:
            hdf_file: FY4B HDF文件路径
        """
        self.hdf_file = hdf_file
        self.filename = os.path.basename(hdf_file)
        self._load_calibration_data()
    
    def _load_calibration_data(self):
        """加载定标数据"""
        with h5py.File(self.hdf_file, 'r') as f:
            # 加载定标系数 (Scale, Offset)
            self.cal_coef = f['Calibration/CALIBRATION_COEF(SCALE+OFFSET)'][()]
            
            # 加载查找表
            self.cal_tables = {}
            for i in range(1, 9):
                ch_name = f'Channel{i:02d}'
                self.cal_tables[ch_name] = f[f'Calibration/CAL{ch_name}'][()]
            
            # 加载太阳辐照度（用于反射率计算）
            self.esun = f['Calibration/ESUN'][()]
            
            # 加载地球太阳距离比
            self.earth_sun_dist_ratio = f.attrs['Earth/Sun Distance Ratio'][0]
            
            # 加载质量标志
            self.cal_quality_flag = f['QA/CalQualityFlag'][()]
            self.l1_quality_flag = f['QA/L1QualityFlag'][()]
            
            # 保存文件元数据
            self.file_attrs = dict(f.attrs)
    
    def get_scale_offset(self, channel: str) -> Tuple[float, float]:
        """
        获取指定波段的定标系数
        
        Args:
            channel: 波段名称, 如 'Channel01'
        
        Returns:
            (scale, offset) 元组
        """
        ch_idx = int(channel.replace('Channel', '')) - 1
        return self.cal_coef[ch_idx, 0], self.cal_coef[ch_idx, 1]
    
    def calibrate_with_lut(self, channel: str, dn_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        使用查找表进行定标（推荐方法）
        
        Args:
            channel: 波段名称, 如 'Channel01'
            dn_data: DN值数组, 如果为None则从文件中读取
        
        Returns:
            定标后的物理量数组
        """
        # 读取DN数据
        if dn_data is None:
            with h5py.File(self.hdf_file, 'r') as f:
                dn_data = f[f'Data/NOM{channel}'][()]
        
        # 获取查找表
        cal_table = self.cal_tables[channel]
        
        # 处理填充值
        fill_mask = (dn_data == self.FILL_VALUE_DN)
        
        # DN值裁剪到查找表范围 (0-4095)
        dn_clipped = np.clip(dn_data, 0, len(cal_table) - 1)
        
        # 查找表定标
        phys_data = cal_table[dn_clipped]
        
        # 设置填充值为nan
        phys_data = phys_data.astype(np.float32)
        phys_data[fill_mask] = np.nan
        
        return phys_data
    
    def calibrate_linear(self, channel: str, dn_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        使用线性公式进行定标
        
        Args:
            channel: 波段名称, 如 'Channel01'
            dn_data: DN值数组, 如果为None则从文件中读取
        
        Returns:
            定标后的物理量数组
        """
        # 读取DN数据
        if dn_data is None:
            with h5py.File(self.hdf_file, 'r') as f:
                dn_data = f[f'Data/NOM{channel}'][()]
        
        # 获取定标系数
        scale, offset = self.get_scale_offset(channel)
        
        # 处理填充值
        fill_mask = (dn_data == self.FILL_VALUE_DN)
        
        # 线性定标: 物理量 = DN × Scale + Offset
        phys_data = dn_data.astype(np.float32) * scale + offset
        
        # 设置填充值为nan
        phys_data[fill_mask] = np.nan
        
        return phys_data
    
    def calibrate_all_bands(self, method: str = 'lut') -> Dict[str, np.ndarray]:
        """
        定标所有波段
        
        Args:
            method: 定标方法, 'lut'(查找表) 或 'linear'(线性)
        
        Returns:
            字典, key为波段名, value为定标后的数据
        """
        results = {}
        cal_func = self.calibrate_with_lut if method == 'lut' else self.calibrate_linear
        
        for ch in self.BAND_INFO.keys():
            print(f"  正在处理 {ch} ({self.BAND_INFO[ch]['name']})...")
            results[ch] = cal_func(ch)
        
        return results
    
    def get_band_info(self, channel: str) -> dict:
        """获取波段信息"""
        return self.BAND_INFO.get(channel, {})
    
    def print_calibration_info(self):
        """打印定标信息"""
        print("="*60)
        print(f"文件: {self.filename}")
        print("="*60)
        print("\n【定标系数 (Scale, Offset)】")
        for ch in self.BAND_INFO.keys():
            ch_idx = int(ch.replace('Channel', '')) - 1
            scale, offset = self.cal_coef[ch_idx]
            band_type = self.BAND_INFO[ch]['type']
            print(f"  {ch} ({self.BAND_INFO[ch]['name']}): "
                  f"Scale={scale:.10f}, Offset={offset:.6f} [{band_type}]")
        
        print("\n【查找表范围】")
        for ch in self.BAND_INFO.keys():
            cal_table = self.cal_tables[ch]
            band_type = self.BAND_INFO[ch]['type']
            unit = '反射率' if band_type == 'reflectance' else '亮温(K)'
            print(f"  {ch}: min={cal_table.min():.4f}, max={cal_table.max():.4f} [{unit}]")
        
        print("\n【质量标志】")
        print(f"  CalQualityFlag: {self.cal_quality_flag}")
        print(f"  L1QualityFlag: {self.l1_quality_flag}")


def save_calibrated_data(calibrator: FY4BCalibrator, 
                         output_dir: str,
                         method: str = 'lut',
                         compress: bool = True):
    """
    保存定标后的数据到HDF5文件
    
    Args:
        calibrator: FY4BCalibrator实例
        output_dir: 输出目录
        method: 定标方法
        compress: 是否使用压缩
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = calibrator.filename.replace('_NOM_', '_CAL_')
    output_path = os.path.join(output_dir, base_name)
    
    print(f"\n保存定标数据到: {output_path}")
    
    # 执行定标
    print(f"\n使用{method}方法进行定标...")
    calibrated_data = calibrator.calibrate_all_bands(method=method)
    
    # 保存到HDF5
    with h5py.File(output_path, 'w') as f_out:
        # 创建数据组
        cal_group = f_out.create_group('CalibratedData')
        
        for ch, data in calibrated_data.items():
            band_info = calibrator.get_band_info(ch)
            
            # 创建数据集
            if compress:
                dset = cal_group.create_dataset(
                    ch, data=data, 
                    compression='gzip', 
                    compression_opts=4,
                    chunks=True
                )
            else:
                dset = cal_group.create_dataset(ch, data=data)
            
            # 添加属性
            dset.attrs['band_name'] = band_info['name']
            dset.attrs['wavelength'] = band_info['wavelength']
            dset.attrs['type'] = band_info['type']
            dset.attrs['calibration_method'] = method
            dset.attrs['fill_value'] = np.nan
            
            # 统计信息
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                dset.attrs['min'] = valid_data.min()
                dset.attrs['max'] = valid_data.max()
                dset.attrs['mean'] = valid_data.mean()
                dset.attrs['std'] = valid_data.std()
        
        # 复制原始文件的元数据
        meta_group = f_out.create_group('Metadata')
        with h5py.File(calibrator.hdf_file, 'r') as f_in:
            for key, val in f_in.attrs.items():
                try:
                    meta_group.attrs[key] = val
                except:
                    pass
        
        # 保存定标参数
        calib_group = f_out.create_group('CalibrationInfo')
        calib_group.create_dataset('CALIBRATION_COEF', data=calibrator.cal_coef)
        calib_group.create_dataset('ESUN', data=calibrator.esun)
        calib_group.attrs['method'] = method
    
    print(f"定标数据已保存: {output_path}")
    return output_path


def process_single_file(input_file: str, 
                        output_dir: str,
                        method: str = 'lut',
                        save: bool = True) -> Dict[str, np.ndarray]:
    """
    处理单个FY4B文件
    
    Args:
        input_file: 输入HDF文件路径
        output_dir: 输出目录
        method: 定标方法
        save: 是否保存结果
    
    Returns:
        定标后的数据字典
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {os.path.basename(input_file)}")
    print(f"{'='*60}")
    
    # 创建定标器
    calibrator = FY4BCalibrator(input_file)
    
    # 打印定标信息
    calibrator.print_calibration_info()
    
    # 保存定标数据
    if save:
        save_calibrated_data(calibrator, output_dir, method=method)
    
    return calibrator.calibrate_all_bands(method=method)


def batch_process(input_dir: str,
                  output_dir: str,
                  pattern: str = '*.HDF',
                  method: str = 'lut'):
    """
    批量处理FY4B文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        method: 定标方法
    """
    # 查找所有匹配文件
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    
    print(f"\n找到 {len(files)} 个文件需要处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"定标方法: {method}")
    
    # 处理每个文件
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] ", end='')
        try:
            process_single_file(file_path, output_dir, method=method, save=True)
        except Exception as e:
            print(f"处理失败: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"批量处理完成! 共处理 {len(files)} 个文件")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FY-4B AGRI L1 数据定标工具')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入文件或目录')
    parser.add_argument('--output', '-o', type=str, default='./calibrated',
                        help='输出目录')
    parser.add_argument('--method', '-m', type=str, default='lut',
                        choices=['lut', 'linear'],
                        help='定标方法: lut(查找表) 或 linear(线性)')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='批量处理模式')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process(args.input, args.output, method=args.method)
    else:
        process_single_file(args.input, args.output, method=args.method, save=True)
