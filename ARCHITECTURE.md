# FY4B 超分辨率研究架构设计

## 核心思想：分层实验框架

本架构将 autoresearch 的"单一指标决策"思想扩展到**两层决策空间**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        决策层次结构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Level 1 (Macro)          Level 2 (Micro)           Level 3    │
│   模型间比较        →      模型内优化        →      模型融合    │
│                                                                 │
│   ┌───────────┐           ┌───────────┐                        │
│   │ bicubic   │           │ lr=0.0001 │  保留                  │
│   │ 30.2 dB   │           │ 32.5 dB   │                        │
│   └───────────┘           ├───────────┤                        │
│   ┌───────────┐           │ lr=0.001  │  保留 ✓                │
│   │ srcnn     │           │ 34.1 dB   │                        │
│   │ 32.5 dB   │           ├───────────┤                        │
│   └───────────┘           │ depth=4   │  丢弃                  │
│   ┌───────────┐           │ 33.8 dB   │                        │
│   │ edsr      │           └───────────┘                        │
│   │ 34.1 dB   │←─────────────────────┐                         │
│   └───────────┘                      │                         │
│   ┌───────────┐                      │                         │
│   │ pftsr     │◄─────────────────────┘ 当前最佳                │
│   │ 33.8 dB   │                                                │
│   └───────────┘                                                │
│                                                                 │
│   决策：选edsr入lv2    决策：选lr=0.001                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 与 autoresearch 的对应关系

| autoresearch 原设计 | 本架构对应 | 说明 |
|---------------------|-----------|------|
| `train.py` (单一文件) | `lv2_micro/experiments/*/main.py` | 每个实验一个独立文件 |
| `prepare.py` (固定工具) | `utils.py` + `data/` | 全局共享工具 |
| `program.md` (任务指令) | `program.md` + `ARCHITECTURE.md` | 分层指令 |
| 单一指标 `val_bpb` | `lv1`: `val_psnr` 比较<br>`lv2`: `val_psnr` 优化 | 同指标，不同用途 |
| git commit/reset | `lv2` 内部使用 | 微观层保持 autoresearch 工作流 |
| 永不停止的循环 | `lv2_micro/run_experiment.sh` | 自动化实验脚本 |

## 三层目录详解

### lv1_macro/ 宏观层

**目标**：找出哪个方法最适合 FY4B 数据

**工作方式**：
```bash
# 运行所有方法，在相同测试集上评估
python compare.py --level macro --band CH07

# 输出：results.csv
# | method          | val_psnr | val_ssim | params | time  |
# |-----------------|----------|----------|--------|-------|
# | 01_bicubic      | 30.2     | 0.85     | 0      | 0s    |
# | 02_srcnn        | 32.5     | 0.89     | 200K   | 120s  |
# | 03_edsr         | 34.1     | 0.91     | 1.5M   | 300s  |
# | 04_pftsr        | 33.8     | 0.91     | 2.8M   | 280s  |
```

**决策**：选择 PSNR 最高的方法进入 lv2_micro
```bash
python lv1_macro/select_best.py
# 输出：03_edsr → 写入 lv2_micro/TARGET_METHOD
```

**关键设计**：
- 每个方法独立目录，隔离依赖
- 固定训练预算（如100 epoch），确保可比性
- 不修改方法内部，只调用统一接口

### lv2_micro/ 微观层

**目标**：优化选定方法的超参数和结构

**工作方式**（纯 autoresearch 模式）：
```bash
cd lv2_micro

# 1. 复制目标方法为基线
cp -r ../lv1_macro/methods/03_edsr experiments/20260412_baseline

# 2. 运行基线
python experiments/20260412_baseline/main.py --band CH07
# 记录结果: 34.1 dB → git commit

# 3. 创建新实验分支
git checkout -b experiment/lr_tune
cp -r experiments/20260412_baseline experiments/20260412_lr_0.01
# 修改 experiments/20260412_lr_0.01/main.py 中的学习率

# 4. 运行实验
python experiments/20260412_lr_0.01/main.py --band CH07
# 结果: 34.5 dB → git commit (保留)
# 结果: 33.8 dB → git reset  (丢弃)
```

**实验记录** (`results.tsv`)：
```tsv
commit      val_psnr  memory_gb  status    description
a1b2c3d     34.100    4.5        keep      edsr baseline
b2c3d4e     34.500    4.5        keep      lr=0.01
c3d4e5f     33.800    4.5        discard   lr=0.1 (diverge)
d4e5f6g     34.600    5.2        keep      depth=32
```

**关键设计**：
- 完全遵循 autoresearch 工作流
- 每个实验独立目录，便于回退
- 单一指标决策，永不停止

### lv3_fusion/ 融合层（可选）

**目标**：组合 lv1 中 top-2 或 top-3 的方法

**触发条件**：lv1 中多个方法表现接近且各有优势

## 完整工作流程

```
阶段1：宏观探索（1-2天）
├── 收集所有候选方法
│   ├── 传统：bicubic, bilinear
│   ├── 早期深度学习：SRCNN, VDSR
│   ├── 先进方法：EDSR, RCAN, SwinIR
│   └── 自定义：PFT-SR
│
├── 统一接口适配
│   └── 每个方法实现：train(), eval(), save()
│
├── 运行宏观比较
│   python compare.py --level macro
│
└── 选择最佳方法
    python lv1_macro/select_best.py
    → 输出：03_edsr 进入微观层

阶段2：微观优化（3-5天，autoresearch模式）
├── 建立基线
│   cp ../lv1_macro/methods/03_edsr experiments/baseline
│   python experiments/baseline/main.py
│   git commit -m "edsr baseline: 34.1 dB"
│
├── 自动化实验循环（永不停止）
│   while true:
│       1. 基于当前最佳提出假设
│       2. 创建新实验目录
│       3. 运行实验
│       4. 比较结果
│       5. 好→保留，差→回退
│
└── 记录最优配置
    → 输出：edsr + lr=0.01 + depth=32: 35.2 dB

阶段3：融合创新（可选，1-2天）
├── 如果 lv1 中 top-2 差距 < 0.5 dB
│   例如：edsr 34.1, pftsr 33.8
│
├── 设计融合策略
│   ├── 特征级融合：concat + conv
│   ├── 像素级融合：加权平均
│   └── 自适应融合：learnable weights
│
└── 训练融合模型
    → 目标：> max(edsr, pftsr) + 0.3 dB

阶段4：验证与输出
├── 在 CH08 上验证通用性
├── 生成对比图表
└── 保存最终模型到 results/final_model/
```

## 关键设计决策

### 1. 为什么 lv1 不用 git 迭代？

autoresearch 的 git 迭代适合**单一方法的参数探索**，但 lv1 是**不同方法的结构比较**：
- EDSR 和 SwinIR 的代码结构完全不同
- 修改 EDSR 的学习率对 SwinIR 没有意义
- 方法间是"选择"关系，不是"迭代"关系

### 2. 为什么 lv2 要保持独立目录？

```
experiments/
├── 20260412_baseline/       # 基线版本
├── 20260412_lr_0.01/        # 实验1
├── 20260412_depth_32/       # 实验2
└── 20260412_no_attention/   # 实验3
```

优点：
- 每个实验可独立回退 (`rm -rf`)
- 可同时对比多个实验结果
- 避免 git 历史混乱

### 3. 指标一致性

```
lv1_macro:  val_psnr (越高越好)
lv2_micro:  val_psnr (越高越好)
            ↑ 相同指标，确保决策一致性
```

### 4. 时间预算

| 层次 | 预算 | 目的 |
|------|------|------|
| lv1 | 100 epoch/方法 | 确保收敛，公平比较 |
| lv2 | 100 epoch/实验 | 与基线一致，可比 |

## 文件职责总结

| 文件/目录 | 层级 | 修改权限 | 说明 |
|-----------|------|----------|------|
| `utils.py` | 全局 | 不可修改 | 固定工具函数 |
| `data/` | 全局 | 不可修改 | 统一数据加载 |
| `compare.py` | lv1 | 可修改 | 宏观比较入口 |
| `lv1_macro/methods/` | lv1 | 可添加 | 各方法实现 |
| `lv1_macro/results.csv` | lv1 | 自动生成 | 比较结果 |
| `lv2_micro/TARGET_METHOD` | lv2 | 自动写入 | 当前优化目标 |
| `lv2_micro/experiments/` | lv2 | 可修改 | 实验目录 |
| `lv2_micro/results.tsv` | lv2 | 自动生成 | 实验记录 |
| `lv2_micro/run_experiment.sh` | lv2 | 可修改 | 自动化脚本 |

## 使用示例

### 快速开始

```bash
# 1. 查看当前架构状态
python status.py

# 2. 运行宏观比较（找出最佳方法）
python compare.py --level macro --band CH07

# 3. 进入微观优化（autoresearch模式）
cd lv2_micro
./run_experiment.sh

# 4. 查看最终对比
python visualize.py --all
```

### 添加新方法

```bash
# 1. 创建方法目录
mkdir lv1_macro/methods/06_method_rrdb

# 2. 实现统一接口
cat > lv1_macro/methods/06_method_rrdb/main.py << 'EOF'
#!/usr/bin/env python3
"""RRDB 方法入口"""
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    # 加载模型、训练、评估
    # ... RRDB 实现 ...
    
    result = {
        'method': 'rrdb',
        'band': args.band,
        'val_psnr': 34.2,  # 实际结果
        'val_ssim': 0.92,
        'train_time': 350,
        'params': 1500000,
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()
EOF

# 3. 运行比较
python compare.py --level macro
```

## 参考

- [autoresearch](https://github.com/karpathy/autoresearch) - 微观层设计灵感
- [Papers With Code - Super-Resolution](https://paperswithcode.com/task/image-super-resolution) - 方法选择参考
