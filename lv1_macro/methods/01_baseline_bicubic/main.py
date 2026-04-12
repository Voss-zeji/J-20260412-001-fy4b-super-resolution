#!/usr/bin/env python3
"""
01_baseline_bicubic - 双三次插值基线

最简单的超分辨率方法，使用双三次插值上采样。
作为所有深度学习方法的性能下限基线。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description='Bicubic Baseline for SR')
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=0, help='基线方法无需训练')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001, help='基线方法无需学习率')
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


class BicubicSR:
    """双三次插值超分辨率"""
    def __init__(self, upscale_factor=2):
        self.upscale_factor = upscale_factor
        self.name = "bicubic"

    def __call__(self, lr_image):
        """
        上采样低分辨率图像
        Args:
            lr_image: [B, C, H, W] 低分辨率图像
        Returns:
            sr_image: [B, C, H*upscale, W*upscale] 超分辨率图像
        """
        sr_image = F.interpolate(
            lr_image,
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False
        )
        # 裁剪到有效范围 [-1, 1]
        sr_image = torch.clamp(sr_image, -1, 1)
        return sr_image

    def parameters(self):
        """返回空参数（无训练参数）"""
        return []

    def to(self, device):
        """兼容接口"""
        return self

    def eval(self):
        """兼容接口"""
        pass


def evaluate_bicubic(model, val_loader, device):
    """评估双三次插值性能"""
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    num_samples = 0

    for lr, hr, _ in val_loader:
        lr = lr.to(device)
        hr = hr.to(device)

        # 双三次上采样
        sr = model(lr)

        # 计算指标
        for i in range(sr.size(0)):
            total_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
            total_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
            total_rmse += calculate_rmse(sr[i:i+1], hr[i:i+1])

        num_samples += sr.size(0)

    return {
        'psnr': total_psnr / num_samples,
        'ssim': total_ssim / num_samples,
        'rmse': total_rmse / num_samples
    }


def main():
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("01_baseline_bicubic - 双三次插值基线")
    print("=" * 60)
    print(f"Band: {args.band}")
    print(f"Device: {device}")

    # 创建数据加载器
    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    channel = args.band.replace('CH', 'Channel')

    print(f"\n加载数据...")
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir,
        high_res_dir=high_res_dir,
        channel=channel,
        batch_size=args.batch_size,
        num_workers=4,
        patch_size=64,
        upscale_factor=2
    )

    # 创建模型（双三次插值）
    print("\n初始化双三次插值...")
    model = BicubicSR(upscale_factor=2)
    model = model.to(device)

    # 统计参数量（无参数）
    model_params = 0
    model_size_mb = 0.0

    # 评估（无需训练）
    print("\n评估验证集...")
    start_time = time.time()
    metrics = evaluate_bicubic(model, val_loader, device)
    eval_time = time.time() - start_time

    # 推理速度测试
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time_ms = (time.time() - t0) * 1000

    # 保存结果
    result = {
        "method": "01_baseline_bicubic",
        "band": args.band,
        "val_psnr": round(metrics['psnr'], 2),
        "val_ssim": round(metrics['ssim'], 4),
        "val_rmse": round(metrics['rmse'], 4),
        "train_time": round(eval_time, 2),
        "train_epochs": 0,
        "model_params": model_params,
        "model_size_mb": model_size_mb,
        "inference_time_ms": round(inference_time_ms, 2),
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print(f"val_psnr: {result['val_psnr']:.2f} dB")
    print(f"val_ssim: {result['val_ssim']:.4f}")
    print(f"inference_time: {result['inference_time_ms']:.2f} ms")
    print(f"结果已保存: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
