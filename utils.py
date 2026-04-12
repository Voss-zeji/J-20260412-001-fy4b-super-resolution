"""
工具函数 - FY-4B 超分辨率研究

包含:
- 评估指标 (PSNR, SSIM, RMSE, MAE)
- 检查点管理 (保存/加载模型)
- 可视化工具
- 图像处理工具
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from math import log10
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt


# ==================== 评估指标 ====================

def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)

    Args:
        img1: 预测图像 (torch.Tensor 或 numpy.ndarray)
        img2: 目标图像
        max_val: 像素最大值 (归一化图像通常为1.0)

    Returns:
        psnr: PSNR值 (dB)
    """
    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * log10(max_val) - 10 * log10(mse.item())
    else:
        # numpy 版本
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    计算SSIM (Structural Similarity Index)

    Args:
        img1: 预测图像 [B, C, H, W] (torch.Tensor)
        img2: 目标图像 [B, C, H, W]
        window_size: 滑动窗口大小
        size_average: 是否对所有通道求平均

    Returns:
        ssim: SSIM值
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 创建高斯窗口
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()

    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)

    window = window_2d.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    # 计算SSIM
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_rmse(pred, target):
    """计算RMSE (Root Mean Square Error)"""
    if isinstance(pred, torch.Tensor):
        mse = torch.mean((pred - target) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()
    else:
        mse = np.mean((pred - target) ** 2)
        return np.sqrt(mse)


def calculate_mae(pred, target):
    """计算MAE (Mean Absolute Error)"""
    if isinstance(pred, torch.Tensor):
        mae = torch.mean(torch.abs(pred - target))
        return mae.item()
    else:
        return np.mean(np.abs(pred - target))


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算峰值信噪比 (PSNR) - numpy版本

    Args:
        img1: 第一张图像
        img2: 第二张图像
        max_val: 最大像素值

    Returns:
        PSNR 值 (dB)
    """
    return calculate_psnr(img1, img2, max_val)


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算结构相似性指数 (SSIM) - numpy版本

    Args:
        img1: 第一张图像
        img2: 第二张图像

    Returns:
        SSIM 值
    """
    from skimage.metrics import structural_similarity

    if len(img1.shape) == 3:
        return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)
    else:
        return structural_similarity(img1, img2, data_range=1.0)


# ==================== 检查点管理 ====================

def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str = 'checkpoint.pth', is_best: bool = False):
    """
    保存模型检查点

    Args:
        state: 包含模型状态、优化器状态等的字典
        save_dir: 保存目录
        filename: 文件名
        is_best: 是否为最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_path)
        print(f"保存最佳模型到: {best_path}")

    print(f"保存检查点: {filepath}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, device: str = 'cuda'):
    """
    加载模型检查点

    Args:
        checkpoint_path: 检查点文件路径
        model: 模型实例
        optimizer: 优化器实例 (可选)
        device: 设备

    Returns:
        start_epoch: 开始的epoch
        best_psnr: 最佳PSNR值
    """
    if not os.path.exists(checkpoint_path):
        print(f"警告: 检查点文件不存在: {checkpoint_path}")
        return 0, 0.0

    print(f"加载检查点: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    best_psnr = checkpoint.get('best_psnr', 0.0)

    print(f"从epoch {start_epoch}恢复训练, 最佳PSNR: {best_psnr:.2f}")

    return start_epoch, best_psnr


def find_last_checkpoint(save_dir: str, pattern: str = 'checkpoint_epoch_*.pth') -> Optional[str]:
    """
    查找最新的检查点文件

    Args:
        save_dir: 保存目录
        pattern: 文件名模式

    Returns:
        last_checkpoint: 最新检查点的路径，如果没有则返回None
    """
    checkpoint_files = glob.glob(os.path.join(save_dir, pattern))

    if not checkpoint_files:
        return None

    # 按修改时间排序
    checkpoint_files.sort(key=os.path.getmtime)

    return checkpoint_files[-1]


def cleanup_old_checkpoints(save_dir: str, keep_last: int = 5):
    """
    清理旧的检查点文件，只保留最近N个

    Args:
        save_dir: 保存目录
        keep_last: 保留的检查点数量
    """
    checkpoint_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))

    if len(checkpoint_files) <= keep_last:
        return

    # 按修改时间排序
    checkpoint_files.sort(key=os.path.getmtime)

    # 删除旧的检查点
    for old_file in checkpoint_files[:-keep_last]:
        try:
            os.remove(old_file)
            print(f"删除旧检查点: {old_file}")
        except Exception as e:
            print(f"删除检查点失败 {old_file}: {e}")


# ==================== 可视化工具 ====================

def denormalize(tensor, min_val=150.0, max_val=350.0):
    """
    反归一化: 将[-1, 1]范围的数据恢复到原始范围

    Args:
        tensor: 归一化后的张量
        min_val: 原始最小值
        max_val: 原始最大值

    Returns:
        denormalized: 反归一化后的数据
    """
    tensor = (tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
    tensor = tensor * (max_val - min_val) + min_val  # [0, 1] -> [min_val, max_val]
    return tensor


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    归一化图像数据到 [0, 1] 范围

    Args:
        image: 输入图像数组

    Returns:
        归一化后的图像
    """
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def save_image(tensor, save_path: str, title: Optional[str] = None, cmap: str = 'jet', vmin=None, vmax=None):
    """
    保存张量为图像

    Args:
        tensor: 输入张量 [H, W] 或 [C, H, W] 或 [B, C, H, W]
        save_path: 保存路径
        title: 图像标题
        cmap: 颜色映射
        vmin, vmax: 颜色范围
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换为numpy
    if isinstance(tensor, np.ndarray):
        img = tensor
    else:
        img = tensor.detach().cpu().numpy()

    # 处理不同维度
    if img.ndim == 4:  # [B, C, H, W]
        img = img[0]  # 取第一个batch
    if img.ndim == 3:  # [C, H, W]
        img = img[0] if img.shape[0] in [1, 3] else img  # 取第一个通道或RGB

    # 确保是2D
    img = img.squeeze()

    # 绘制图像
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"保存图像: {save_path}")


def visualize_results(lr, sr, hr, save_dir: str, epoch: int, idx: int = 0, channel_names: Optional[list] = None):
    """
    可视化超分辨率结果

    Args:
        lr: 低分辨率图像 [B, C, H, W]
        sr: 超分辨率图像 [B, C, H, W]
        hr: 高分辨率图像 [B, C, H, W]
        save_dir: 保存目录
        epoch: 当前epoch
        idx: 样本索引
        channel_names: 通道名称列表
    """
    from scipy.ndimage import zoom

    os.makedirs(save_dir, exist_ok=True)

    if channel_names is None:
        channel_names = [f'Ch{i}' for i in range(lr.size(1))]

    # 转换为numpy并反归一化
    lr_np = denormalize(lr[idx].detach().cpu()).numpy()
    sr_np = denormalize(sr[idx].detach().cpu()).numpy()
    hr_np = denormalize(hr[idx].detach().cpu()).numpy()

    # 为每个通道创建可视化
    num_channels = lr.size(1)
    num_display = min(num_channels, 8)  # 最多显示8个通道

    fig, axes = plt.subplots(num_display, 3, figsize=(12, 4*num_display))

    for ch in range(num_display):
        # 计算显示范围
        vmin = min(lr_np[ch].min(), sr_np[ch].min(), hr_np[ch].min())
        vmax = max(lr_np[ch].max(), sr_np[ch].max(), hr_np[ch].max())

        if num_display == 1:
            ax_row = [axes[0], axes[1], axes[2]]
        else:
            ax_row = axes[ch]

        # 低分辨率 (双三次插值上采样以便比较)
        lr_upscaled = zoom(lr_np[ch], sr_np.shape[-1] / lr_np.shape[-1], order=1)
        ax_row[0].imshow(lr_upscaled, cmap='jet', vmin=vmin, vmax=vmax)
        ax_row[0].set_title(f'{channel_names[ch]} - LR (Bicubic)')
        ax_row[0].axis('off')

        # 超分辨率结果
        ax_row[1].imshow(sr_np[ch], cmap='jet', vmin=vmin, vmax=vmax)
        ax_row[1].set_title(f'{channel_names[ch]} - SR (Ours)')
        ax_row[1].axis('off')

        # 高分辨率真值
        ax_row[2].imshow(hr_np[ch], cmap='jet', vmin=vmin, vmax=vmax)
        ax_row[2].set_title(f'{channel_names[ch]} - HR (GT)')
        ax_row[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'epoch_{epoch}_sample_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"保存可视化结果: {save_path}")


def plot_training_curves(history: Dict[str, list], save_path: str):
    """
    绘制训练曲线

    Args:
        history: 训练历史字典，包含loss、psnr、ssim等
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['loss']) + 1)

    # 获取验证发生的epoch（如果有记录）
    if 'val_epochs' in history and len(history['val_epochs']) > 0:
        val_epochs = history['val_epochs']
    else:
        # 兼容旧版本，假设验证每个epoch都进行
        val_epochs = list(epochs)

    # Loss曲线
    axes[0].plot(epochs, history['loss'], 'b-', label='Train Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(val_epochs[:len(history['val_loss'])], history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # PSNR曲线
    if 'val_psnr' in history and len(history['val_psnr']) > 0:
        axes[1].plot(val_epochs[:len(history['val_psnr'])], history['val_psnr'], 'g-', label='PSNR')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('Validation PSNR')
        axes[1].legend()
        axes[1].grid(True)

    # SSIM曲线
    if 'val_ssim' in history and len(history['val_ssim']) > 0:
        axes[2].plot(val_epochs[:len(history['val_ssim'])], history['val_ssim'], 'm-', label='SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('SSIM')
        axes[2].set_title('Validation SSIM')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"保存训练曲线: {save_path}")


# ==================== 图像处理工具 ====================

def create_lr_hr_pair(hr_image: np.ndarray, scale_factor: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    从高分辨率图像创建低分辨率-高分辨率图像对

    Args:
        hr_image: 高分辨率图像
        scale_factor: 下采样倍数

    Returns:
        (lr_image, hr_image) 图像对
    """
    from scipy.ndimage import zoom

    h, w = hr_image.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor

    # 下采样创建低分辨率图像
    lr_image = zoom(hr_image, (lr_h/h, lr_w/w, 1) if len(hr_image.shape) == 3
                    else (lr_h/h, lr_w/w), order=1)

    return lr_image, hr_image


# ==================== 模型评估工具 ====================

def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    """
    评估模型性能

    Args:
        model: 超分辨率模型
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        metrics: 包含PSNR和SSIM平均值的字典
    """
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (lr, hr, _) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            # 前向传播
            sr = model(lr)

            # 计算指标
            batch_psnr = 0.0
            batch_ssim = 0.0

            for i in range(sr.size(0)):
                batch_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
                batch_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])

            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_samples += sr.size(0)

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }


def calculate_channel_metrics(pred: torch.Tensor, target: torch.Tensor, channel_names: list) -> Dict[str, Dict[str, float]]:
    """
    计算每个通道的评估指标

    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]
        channel_names: 通道名称列表

    Returns:
        channel_metrics: 每个通道的指标字典
    """
    channel_metrics = {}

    for i, name in enumerate(channel_names):
        ch_pred = pred[:, i:i+1]
        ch_target = target[:, i:i+1]

        psnr_val = calculate_psnr(ch_pred, ch_target)
        ssim_val = calculate_ssim(ch_pred, ch_target)

        channel_metrics[name] = {
            'psnr': psnr_val,
            'ssim': ssim_val
        }

    return channel_metrics
