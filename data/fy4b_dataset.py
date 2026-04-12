# -*- coding: utf-8 -*-
"""
FY-4B卫星数据集类
支持加载低-高分辨率数据对进行超分辨率训练

数据路径格式:
- /root/autodl-tmp/FY-4B/calibration/2000M/CH07/  (高分辨率)
- /root/autodl-tmp/FY-4B/calibration/4000M/CH07/  (低分辨率)
"""

import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import h5py


class FY4BDataset(Dataset):
    """
    FY-4B卫星超分辨率数据集
    
    支持的数据对:
    - 2km-4km: 4000M -> 2000M，上采样因子 2x
    
    参数:
        low_res_dir: 低分辨率数据目录 (如 /root/autodl-tmp/FY-4B/calibration/4000M/CH07)
        high_res_dir: 高分辨率数据目录 (如 /root/autodl-tmp/FY-4B/calibration/2000M/CH07)
        channel: 使用的通道名称, 如 'Channel07' 或 'Channel08'
        patch_size: 裁剪的图像块大小 (低分辨率空间)
        upscale_factor: 上采样因子 (默认2: 4km->2km)
        mode: 'train' 或 'val'
    """
    
    # 支持的通道列表
    SUPPORTED_CHANNELS = ['Channel07', 'Channel08']
    
    def __init__(
        self,
        low_res_dir=None,
        high_res_dir=None,
        low_res_file=None,
        high_res_file=None,
        channel='Channel07',
        patch_size=64,
        upscale_factor=2,
        mode='train'
    ):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.low_res_file = low_res_file
        self.high_res_file = high_res_file
        self.channel = channel
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.mode = mode
        
        # 验证通道
        if channel not in self.SUPPORTED_CHANNELS:
            raise ValueError(f"不支持的通道: {channel}，支持的通道: {self.SUPPORTED_CHANNELS}")
        
        # 数据对列表
        self.data_pairs = []
        
        # 扫描数据对
        self._scan_data_pairs()
        
        print(f"[{mode}] 找到 {len(self.data_pairs)} 个数据对")
        print(f"使用通道: {self.channel}")
    
    def _scan_data_pairs(self):
        """
        扫描并匹配低-高分辨率数据对
        从定标后的数据目录中扫描匹配的文件
        """
        if self.low_res_dir and self.high_res_dir:
            # 检查目录是否存在
            if not os.path.exists(self.low_res_dir):
                raise FileNotFoundError(f"低分辨率数据目录不存在: {self.low_res_dir}")
            if not os.path.exists(self.high_res_dir):
                raise FileNotFoundError(f"高分辨率数据目录不存在: {self.high_res_dir}")
            
            # 实际数据扫描逻辑
            low_res_files = self._get_hdf_files(self.low_res_dir)
            high_res_files = self._get_hdf_files(self.high_res_dir)
            
            print(f"  低分辨率文件数: {len(low_res_files)}")
            print(f"  高分辨率文件数: {len(high_res_files)}")
            
            # 根据时间戳匹配数据对
            for low_file in low_res_files:
                timestamp = self._extract_timestamp(low_file)
                if timestamp:
                    # 在high_res_files中查找匹配的时间戳
                    matching_high = self._find_matching_file(timestamp, high_res_files)
                    if matching_high:
                        self.data_pairs.append({
                            'low': low_file,
                            'high': matching_high,
                            'timestamp': timestamp
                        })
        else:
            # 使用占位符进行测试
            print(f"使用占位符文件: {self.low_res_file} -> {self.high_res_file}")
            # 创建虚拟数据对用于测试
            self.data_pairs = [
                {
                    'low': self.low_res_file,
                    'high': self.high_res_file,
                    'timestamp': f'test_{i}'
                }
                for i in range(10)  # 创建10个虚拟样本
            ]
    
    def _get_hdf_files(self, directory):
        """获取目录中的所有HDF文件"""
        if not os.path.exists(directory):
            return []
        # 匹配定标后的文件名格式 (包含 CAL)
        # 支持两种格式:
        # 1. FY4B-*_AGRI--_N_DISK_*_CAL_*.HDF (原始格式)
        # 2. FY4B_CH07_CAL_*.HDF (新简化格式)
        patterns = [
            os.path.join(directory, 'FY4B-*_AGRI--_N_DISK_*.HDF'),
            os.path.join(directory, 'FY4B_CH*_CAL_*.HDF'),
        ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        # 去重并过滤掉临时文件
        files = list(set(f for f in files if not f.endswith('.tmp')))
        return sorted(files)
    
    def _extract_timestamp(self, filepath):
        """从文件名中提取时间戳"""
        basename = os.path.basename(filepath)
        
        # 尝试匹配新格式: FY4B_CH07_CAL_20250302000000.HDF
        pattern_new = r'FY4B_CH\d+_CAL_(\d{14})\.HDF'
        match = re.search(pattern_new, basename)
        if match:
            return match.group(1)  # 返回时间戳
        
        # 尝试匹配定标后的文件名 (CAL): CAL_20250322183000_20250322184459
        pattern_cal = r'CAL_(\d{14})_(\d{14})'
        match = re.search(pattern_cal, basename)
        if match:
            return match.group(1)  # 返回开始时间
        
        # 尝试匹配原始文件名 (NOM): NOM_20250322183000_20250322184459
        pattern_nom = r'NOM_(\d{14})_(\d{14})'
        match = re.search(pattern_nom, basename)
        if match:
            return match.group(1)
        
        return None
    
    def _find_matching_file(self, timestamp, high_res_files):
        """根据时间戳查找匹配的高分辨率文件"""
        for high_file in high_res_files:
            high_timestamp = self._extract_timestamp(high_file)
            if high_timestamp == timestamp:
                return high_file
        return None
    
    def _load_hdf_data(self, filepath, channel):
        """
        从定标后的HDF文件加载指定通道的数据
        
        定标后文件结构:
        - 根目录下有 Channel07 或 Channel08 数据集
        - 数据形状: 2000M为(5496, 5496)，4000M为(2748, 2748)
        """
        # 检查是否为占位符模式
        if not os.path.exists(filepath):
            # 临时返回随机数据
            if '4000M' in filepath or 'lowResfile' in filepath:
                size = 64
            else:
                size = 128
            np.random.seed(hash(filepath + channel) % 2**32)
            data = np.random.randn(size, size).astype(np.float32) * 50 + 300
            return data
        
        # 从实际HDF文件读取数据
        # 支持两种key格式: Channel07 或 CH07
        channel_key_map = {
            'Channel07': 'CH07',
            'Channel08': 'CH08'
        }
        actual_channel = channel_key_map.get(channel, channel)

        with h5py.File(filepath, 'r') as f:
            if actual_channel not in f:
                raise KeyError(f"通道 {channel} 不在文件 {filepath} 中")
            
            data = f[actual_channel][()]
            
            # 处理NaN值（使用邻近有效值填充或替换为平均值）
            if np.any(~np.isfinite(data)):
                nan_mask = ~np.isfinite(data)
                # 使用全局平均值填充NaN
                valid_mean = np.nanmean(data)
                data = np.where(nan_mask, valid_mean, data)
            
            return data.astype(np.float32)
    
    def _crop_patch(self, lr_img, hr_img):
        """随机裁剪图像块"""
        h, w = lr_img.shape
        
        if self.mode == 'train':
            # 随机裁剪
            h_start = np.random.randint(0, h - self.patch_size + 1)
            w_start = np.random.randint(0, w - self.patch_size + 1)
        else:
            # 验证模式下从中心裁剪
            h_start = (h - self.patch_size) // 2
            w_start = (w - self.patch_size) // 2
        
        # 低分辨率裁剪
        lr_patch = lr_img[h_start:h_start+self.patch_size, 
                         w_start:w_start+self.patch_size]
        
        # 高分辨率对应区域 (upscale_factor倍)
        hr_h_start = h_start * self.upscale_factor
        hr_w_start = w_start * self.upscale_factor
        hr_patch_size = self.patch_size * self.upscale_factor
        hr_patch = hr_img[hr_h_start:hr_h_start+hr_patch_size,
                         hr_w_start:hr_w_start+hr_patch_size]
        
        return lr_patch, hr_patch
    
    def _augment(self, lr, hr):
        """数据增强"""
        if self.mode != 'train':
            return lr, hr
        
        # 随机水平翻转
        if np.random.random() < 0.5:
            lr = np.fliplr(lr).copy()
            hr = np.fliplr(hr).copy()
        
        # 随机垂直翻转
        if np.random.random() < 0.5:
            lr = np.flipud(lr).copy()
            hr = np.flipud(hr).copy()
        
        # 随机旋转90度
        if np.random.random() < 0.5:
            k = np.random.randint(1, 4)
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()
        
        return lr, hr
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        获取一个数据样本
        
        返回:
            lr_img: 低分辨率图像 [1, H, W] (单通道)
            hr_img: 高分辨率图像 [1, H, W] (单通道)
            info: 附加信息字典
        """
        pair = self.data_pairs[idx]
        
        # 加载单通道数据
        lr_data = self._load_hdf_data(pair['low'], self.channel)
        hr_data = self._load_hdf_data(pair['high'], self.channel)
        
        # 裁剪图像块
        lr_patch, hr_patch = self._crop_patch(lr_data, hr_data)
        
        # 数据增强
        lr_patch, hr_patch = self._augment(lr_patch, hr_patch)
        
        # 转换为张量 [1, H, W]
        lr_img = torch.from_numpy(lr_patch[np.newaxis, ...])
        hr_img = torch.from_numpy(hr_patch[np.newaxis, ...])
        
        # 归一化
        lr_img = self._normalize(lr_img)
        hr_img = self._normalize(hr_img)
        
        info = {
            'timestamp': pair['timestamp'],
            'channel': self.channel,
            'low_res_file': pair['low'],
            'high_res_file': pair['high']
        }
        
        return lr_img, hr_img, info
    
    def _normalize(self, img):
        """归一化到[-1, 1]范围"""
        # 假设FY-4B亮温数据范围约为 150K - 350K
        min_val = 150.0
        max_val = 350.0
        img = (img - min_val) / (max_val - min_val)
        img = img * 2 - 1  # 转换到[-1, 1]
        return img


def create_dataloaders(
    low_res_dir=None,
    high_res_dir=None,
    low_res_file=None,
    high_res_file=None,
    batch_size=16,
    num_workers=4,
    patch_size=64,
    upscale_factor=2,
    channel='Channel07'
):
    """
    创建训练和验证数据加载器
    
    参数:
        low_res_dir: 低分辨率数据目录 (如 /root/autodl-tmp/FY-4B/calibration/4000M/CH07)
        high_res_dir: 高分辨率数据目录 (如 /root/autodl-tmp/FY-4B/calibration/2000M/CH07)
        low_res_file: 低分辨率数据文件占位符 (用于测试)
        high_res_file: 高分辨率数据文件占位符 (用于测试)
        batch_size: 批大小
        num_workers: 数据加载线程数
        patch_size: 图像块大小 (低分辨率空间)
        upscale_factor: 上采样因子
        channel: 使用的通道名称 (Channel07 或 Channel08)
    
    返回:
        train_loader, val_loader
    """
    
    # 训练数据集
    train_dataset = FY4BDataset(
        low_res_dir=low_res_dir,
        high_res_dir=high_res_dir,
        low_res_file=low_res_file,
        high_res_file=high_res_file,
        channel=channel,
        patch_size=patch_size,
        upscale_factor=upscale_factor,
        mode='train'
    )
    
    # 验证数据集 (使用不同的数据或分割)
    val_dataset = FY4BDataset(
        low_res_dir=low_res_dir,
        high_res_dir=high_res_dir,
        low_res_file=low_res_file,
        high_res_file=high_res_file,
        channel=channel,
        patch_size=patch_size,
        upscale_factor=upscale_factor,
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # 测试数据集
    print("=" * 50)
    print("测试FY4B数据集")
    print("=" * 50)
    
    # 测试1: 使用占位符
    print("\n--- 测试占位符模式 ---")
    dataset = FY4BDataset(
        low_res_file="lowResfile1",
        high_res_file="highResfile2",
        channel='Channel07',
        patch_size=64,
        upscale_factor=2,
        mode='train'
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 获取一个样本
    lr, hr, info = dataset[0]
    print(f"\n低分辨率图像形状: {lr.shape}")
    print(f"高分辨率图像形状: {hr.shape}")
    print(f"信息: {info}")
    
    # 测试2: 使用实际数据路径 (如果存在)
    print("\n--- 测试实际数据模式 ---")
    low_res_dir = '/root/autodl-tmp/FY-4B/calibration/4000M/CH07'
    high_res_dir = '/root/autodl-tmp/FY-4B/calibration/2000M/CH07'
    
    if os.path.exists(low_res_dir) and os.path.exists(high_res_dir):
        dataset_real = FY4BDataset(
            low_res_dir=low_res_dir,
            high_res_dir=high_res_dir,
            channel='Channel07',
            patch_size=64,
            upscale_factor=2,
            mode='train'
        )
        
        print(f"\n数据集大小: {len(dataset_real)}")
        
        if len(dataset_real) > 0:
            lr, hr, info = dataset_real[0]
            print(f"\n低分辨率图像形状: {lr.shape}")
            print(f"高分辨率图像形状: {hr.shape}")
            print(f"信息: {info}")
    else:
        print(f"实际数据路径不存在，跳过测试")
