# lv1_macro - 宏观层：方法间比较

## 目标

在 FY4B 数据集上横向比较不同的超分辨率方法，找出最适合的基础架构。

## 核心原则

```
┌─────────────────────────────────────────────────────┐
│                    公平比较原则                       │
├─────────────────────────────────────────────────────┤
│ 1. 相同数据：所有方法使用相同的 train/val 划分        │
│ 2. 相同预算：固定 100 epoch（或固定时间）            │
│ 3. 相同指标：统一使用 val_psnr 作为评价标准          │
│ 4. 不修改内部：只调用统一接口，不做方法内部优化      │
└─────────────────────────────────────────────────────┘
```

## 目录结构

```
lv1_macro/
├── README.md                  # 本文件
├── methods/                   # 各方法独立目录
│   ├── 01_baseline_bicubic/   # 双三次插值基线
│   ├── 02_baseline_srcnn/     # SRCNN (2015)
│   ├── 03_method_edsr/        # EDSR (2017)
│   ├── 04_method_pftsr/       # PFT-SR (自定义)
│   ├── 05_method_swinir/      # SwinIR (2021)
│   └── __init__.py            # 统一接口定义
├── results.csv                # 比较结果（自动生成）
└── select_best.py             # 选择最佳方法
```

## 方法命名规范

```
XX_category_name/

XX:     两位数字，表示优先级/顺序（01, 02, 03...）
category: 类别
  - baseline: 基线方法（简单、快速）
  - method: 先进方法（复杂、耗时）
  - custom: 自定义方法
name:   方法名称（小写，下划线分隔）

示例：
  01_baseline_bicubic
  02_baseline_srcnn
  03_method_edsr
  04_method_pftsr
```

## 统一接口规范

每个方法必须提供 `main.py`，支持以下命令行参数：

```bash
python main.py \
  --band CH07 \           # 通道选择（CH07 或 CH08）
  --epochs 100 \          # 训练轮数（默认100）
  --batch-size 16 \       # 批次大小（可选）
  --lr 0.001 \            # 学习率（可选）
  --output result.json    # 结果输出文件（必须）
```

### 输出格式（result.json）

```json
{
  "method": "edsr",
  "band": "CH07",
  "val_psnr": 34.15,
  "val_ssim": 0.9123,
  "val_rmse": 15.23,
  "train_time": 285.5,
  "train_epochs": 100,
  "model_params": 1512934,
  "model_size_mb": 6.1,
  "inference_time_ms": 12.5,
  "status": "success",
  "timestamp": "2026-04-12T10:30:00"
}
```

### 最小实现模板

```python
#!/usr/bin/env python3
"""方法名称 - 简要描述"""

import argparse
import json
import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.fy4b_dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', type=str, required=True, choices=['CH07', 'CH08'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


def create_model():
    """创建模型"""
    # TODO: 实现模型
    pass


def train(model, train_loader, val_loader, epochs, lr, device):
    """训练模型"""
    # TODO: 实现训练
    pass


def evaluate(model, val_loader, device):
    """评估模型"""
    # TODO: 实现评估
    pass


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据加载器
    low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/4000M/{args.band}"
    high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/2000M/{args.band}"
    channel = args.band.replace('CH', 'Channel')
    
    train_loader, val_loader = create_dataloaders(
        low_res_dir=low_res_dir,
        high_res_dir=high_res_dir,
        channel=channel,
        batch_size=args.batch_size,
        num_workers=4,
        patch_size=64,
        upscale_factor=2
    )
    
    # 创建模型
    model = create_model().to(device)
    
    # 统计参数量
    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_params * 4 / (1024 ** 2)  # FP32
    
    # 训练
    start_time = time.time()
    model, train_history = train(model, train_loader, val_loader, 
                                  args.epochs, args.lr, device)
    train_time = time.time() - start_time
    
    # 评估
    metrics = evaluate(model, val_loader, device)
    
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
        "method": Path(__file__).parent.name,
        "band": args.band,
        "val_psnr": metrics['psnr'],
        "val_ssim": metrics['ssim'],
        "val_rmse": metrics.get('rmse', 0),
        "train_time": round(train_time, 2),
        "train_epochs": args.epochs,
        "model_params": model_params,
        "model_size_mb": round(model_size_mb, 2),
        "inference_time_ms": round(inference_time_ms, 2),
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存: {args.output}")
    print(f"val_psnr: {result['val_psnr']:.2f} dB")


if __name__ == '__main__':
    main()
```

## 使用方法

### 运行单个方法

```bash
cd lv1_macro/methods/03_method_edsr
python main.py --band CH07 --output result.json
```

### 运行所有方法比较

```bash
# 从项目根目录
python compare.py --level macro --band CH07

# 输出：lv1_macro/results.csv
```

### 选择最佳方法

```bash
python lv1_macro/select_best.py

# 输出示例：
# 最佳方法: 03_method_edsr
# val_psnr: 34.15 dB
# 已写入 lv2_micro/TARGET_METHOD
```

## 结果解读

### results.csv 格式

```csv
rank,method,band,val_psnr,val_ssim,model_params,train_time,status
1,03_method_edsr,CH07,34.15,0.9123,1512934,285.5,success
2,04_method_pftsr,CH07,33.82,0.9087,2847291,278.3,success
3,02_baseline_srcnn,CH07,32.51,0.8912,198432,125.6,success
4,01_baseline_bicubic,CH07,30.12,0.8523,0,0.5,success
5,05_method_swinir,CH07,0,0,0,0,failed
```

### 决策标准

1. **主要指标**: `val_psnr`（越高越好）
2. **效率考虑**: 相同 PSNR 下，选择参数量小、训练快的方法
3. **失败处理**: `status=failed` 的方法不进入 lv2_micro

## 添加新方法

1. 创建目录：`mkdir methods/XX_category_name`
2. 复制模板：`cp methods/03_method_edsr/main.py methods/XX_category_name/`
3. 实现模型和训练逻辑
4. 测试运行：`python main.py --band CH07 --output test.json`
5. 加入比较：`python compare.py --level macro`

## 注意事项

1. **不要在此层做超参调优** - 那是 lv2_micro 的工作
2. **保持公平** - 所有方法使用相同的训练预算
3. **记录失败** - 即使方法失败也要记录原因（OOM、收敛问题等）
4. **版本锁定** - 进入 lv2 后，该方法的代码作为基线不再大幅修改

## 参考

- 基线方法：https://github.com/krishnaw14/SuperResolution-Models
- 先进方法：https://github.com/xinntao/BasicSR
- SwinIR：https://github.com/JingyunLiang/SwinIR
