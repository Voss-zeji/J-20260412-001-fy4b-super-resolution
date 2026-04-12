# lv2_micro - 微观层：模型内优化（autoresearch 模式）

## 目标

在 lv1_macro 选定的最佳方法基础上，通过系统化实验找到最优的超参数和结构配置。

## 核心思想：纯 autoresearch 模式

```
┌─────────────────────────────────────────────────────────────────┐
│                     autoresearch 工作流                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   基线实验 ──► 提出假设 ──► 修改代码 ──► 运行实验 ──► 评估结果    │
│       ▲                                      │                  │
│       │                                      │                  │
│       └──────── 好则保留，差则回退 ───────────┘                  │
│                                                                 │
│   决策原则：单一指标 val_psnr（越高越好）                         │
│   决策动作：git commit 或 git reset / rm -rf                     │
│   运行模式：永不停止，直到人为中断                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
lv2_micro/
├── README.md                    # 本文件
├── TARGET_METHOD                # 文件：记录当前优化目标
├── experiments/                 # 实验目录
│   ├── 20260412_baseline/       # 基线实验（必须保留）
│   │   ├── main.py
│   │   ├── result.json
│   │   └── run.log
│   ├── 20260412_lr_0.0005/      # 学习率实验示例
│   ├── 20260412_lr_0.002/       # 学习率实验示例
│   ├── 20260412_depth_2/        # 深度实验示例
│   └── ...                      # 更多实验
├── results.tsv                  # 实验记录（autoresearch格式）
├── run_experiment.sh            # 自动化实验脚本
└── analyze.py                   # 结果分析工具
```

## 工作流程

### 第一步：初始化基线

```bash
cd lv2_micro

# 1. 读取 lv1_macro 选择的目标方法
TARGET=$(cat TARGET_METHOD)  # 例如：03_method_edsr

# 2. 复制为基线
cp -r ../lv1_macro/methods/$TARGET experiments/$(date +%Y%m%d)_baseline

# 3. 运行基线
python experiments/$(date +%Y%m%d)_baseline/main.py \
  --band CH07 \
  --output experiments/$(date +%Y%m%d)_baseline/result.json

# 4. 记录基线到 results.tsv
echo -e "experiment\tval_psnr\tmemory_gb\tstatus\tdescription" > results.tsv
echo -e "20260412_baseline\t34.15\t4.5\tkeep\tedsr baseline from lv1" >> results.tsv
```

### 第二步：实验循环（永不停止）

```bash
while true; do
    # 1. 查看最新结果
    BEST=$(tail -n +2 results.tsv | sort -k2 -nr | head -1)
    BEST_EXP=$(echo $BEST | cut -f1)
    BEST_PSNR=$(echo $BEST | cut -f2)
    
    echo "当前最佳: $BEST_EXP (PSNR: $BEST_PSNR dB)"
    
    # 2. 提出实验假设（手动或自动）
    # 示例：尝试不同学习率
    NEW_EXP="$(date +%Y%m%d)_lr_0.001"
    
    # 3. 创建新实验
    cp -r experiments/$BEST_EXP experiments/$NEW_EXP
    
    # 4. 修改代码（这是你要做的）
    # 例如：修改学习率
    sed -i 's/LR = 0.0001/LR = 0.001/g' experiments/$NEW_EXP/main.py
    
    # 5. 运行实验
    python experiments/$NEW_EXP/main.py \
      --band CH07 \
      --output experiments/$NEW_EXP/result.json \
      2>&1 | tee experiments/$NEW_EXP/run.log
    
    # 6. 提取结果
    NEW_PSNR=$(grep '"val_psnr"' experiments/$NEW_EXP/result.json | grep -o '[0-9.]*')
    
    # 7. 评估并记录
    if (( $(echo "$NEW_PSNR > $BEST_PSNR" | bc -l) )); then
        STATUS="keep"
        echo "✓ 改善: $NEW_PSNR > $BEST_PSNR"
    else
        STATUS="discard"
        echo "✗ 变差: $NEW_PSNR <= $BEST_PSNR"
        # 可选：删除失败的实验
        # rm -rf experiments/$NEW_EXP
    fi
    
    # 记录结果
    echo -e "$NEW_EXP\t$NEW_PSNR\t4.5\t$STATUS\tlr=0.001" >> results.tsv
    
    # 8. 继续下一个实验（永不停止）
    sleep 1
done
```

### 第三步：结果分析

```bash
# 查看所有实验结果
python analyze.py

# 输出示例：
# ┌─────────────────────────┬──────────┬────────┬────────────────┐
# │ Experiment              │ val_psnr │ Status │ Description    │
# ├─────────────────────────┼──────────┼────────┼────────────────┤
# │ 20260412_baseline       │ 34.15    │ keep   │ edsr baseline  │
# │ 20260412_lr_0.001       │ 34.52    │ keep   │ lr=0.001       │
# │ 20260412_lr_0.01        │ 33.80    │ discard│ lr=0.01 diverge│
# │ 20260412_depth_4        │ 34.60    │ keep   │ depth=4        │
# │ ...                     │ ...      │ ...    │ ...            │
# └─────────────────────────┴──────────┴────────┴────────────────┘
```

## 实验记录格式（results.tsv）

```tsv
experiment	val_psnr	memory_gb	status	description
20260412_baseline	34.15	4.5	keep	edsr baseline from lv1
20260412_lr_0.001	34.52	4.5	keep	lr=0.001, +0.37dB
20260412_lr_0.01	33.80	4.5	discard	lr=0.01, diverged
20260412_depth_4	34.60	5.2	keep	depth=4, +0.08dB from best
20260412_no_attention	33.90	4.2	discard	remove CBAM, -0.62dB
```

**字段说明**：
- `experiment`: 实验名称（日期_描述）
- `val_psnr`: 验证集 PSNR（越高越好）
- `memory_gb`: 峰值显存使用
- `status`: `keep`（保留）/ `discard`（丢弃）/ `crash`（崩溃）
- `description`: 简短描述改动内容

## 可调参数清单

在 `main.py` 中通常可以修改的参数：

```python
class Config:
    # 模型架构
    NUM_FEATURES = 64          # 特征维度: 32, 64, 128
    NUM_PFT_BLOCKS = 3         # 块数量: 2, 3, 4, 5
    NUM_RB_PER_BLOCK = 3       # 残差块数: 2, 3, 4
    USE_ATTENTION = True       # 注意力: True, False
    
    # 训练参数
    LR = 0.0001                # 学习率: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
    BATCH_SIZE = 8             # 批次: 4, 8, 16, 32
    NUM_EPOCHS = 100           # 轮数: 50, 100, 200
    
    # 损失函数
    LAMBDA_L1 = 1.0            # L1权重: 0.5, 1.0, 2.0
    LAMBDA_SSIM = 0.5          # SSIM权重: 0.0, 0.5, 1.0
    
    # 正则化
    WEIGHT_DECAY = 0.0001      # 权重衰减: 0, 1e-4, 1e-3
    GRAD_CLIP = 1.0            # 梯度裁剪: 0.5, 1.0, 2.0
```

## 典型实验序列

### 序列1：学习率探索
```
baseline:      lr=0.0001, psnr=34.15  → keep (基线)
exp_lr_0.0005: lr=0.0005, psnr=34.42  → keep (+0.27)
exp_lr_0.001:  lr=0.001,  psnr=34.65  → keep (+0.23) ← 最佳
exp_lr_0.002:  lr=0.002,  psnr=34.30  → discard (-0.35)
exp_lr_0.005:  lr=0.005,  psnr=33.10  → discard (diverge)
结论：最优学习率 0.001
```

### 序列2：模型深度探索
```
基于 lr=0.001 的结果：
exp_depth_2:   blocks=2, psnr=33.80   → discard (-0.85)
exp_depth_3:   blocks=3, psnr=34.65   → keep (基线)
exp_depth_4:   blocks=4, psnr=34.72   → keep (+0.07) ← 最佳
exp_depth_5:   blocks=5, psnr=34.68   → discard (-0.04)
结论：最优深度 4
```

### 序列3：消融实验
```
基于 depth=4, lr=0.001：
exp_no_attn:   no CBAM,   psnr=34.10   → discard (-0.62)
exp_no_ssim:   λ_ssim=0,  psnr=34.50   → discard (-0.22)
exp_l1_only:   λ_ssim=0, λ_l1=2.0, psnr=34.30 → discard
结论：所有组件都有贡献
```

## 决策规则

### 简化准则（来自 autoresearch）

```
改善幅度 vs 代码复杂度：

+0.5 dB, 添加 50 行 hacky 代码 → 不值得（复杂度过高）
+0.5 dB, 删除 20 行代码       → 值得（简化+改善）
+0.0 dB, 删除 30 行代码       → 值得（纯简化收益）
+0.1 dB, 添加 100 行代码      → 不值得（边际效益低）
```

### 早停条件

```python
# 如果连续 N 次实验没有改善，考虑：
# 1. 换其他参数方向
# 2. 尝试组合之前的最佳结果
# 3. 进入 lv3_fusion（如果提升停滞）

PATIENCE = 10  # 连续10次无改善则调整策略
```

## 工具脚本

### run_experiment.sh

```bash
#!/bin/bash
# 自动化实验脚本

set -e

BAND=${1:-CH07}
EXPERIMENT_NAME=${2:-$(date +%Y%m%d)_exp}
DESCRIPTION=${3:-""}

echo "Running experiment: $EXPERIMENT_NAME"
echo "Band: $BAND"
echo "Description: $DESCRIPTION"

# 运行实验
python experiments/$EXPERIMENT_NAME/main.py \
  --band $BAND \
  --output experiments/$EXPERIMENT_NAME/result.json \
  2>&1 | tee experiments/$EXPERIMENT_NAME/run.log

# 提取结果
if [ -f experiments/$EXPERIMENT_NAME/result.json ]; then
    PSNR=$(grep '"val_psnr"' experiments/$EXPERIMENT_NAME/result.json | grep -o '[0-9.]*')
    echo "Result: val_psnr = $PSNR dB"
else
    echo "Error: result.json not found"
    exit 1
fi
```

### analyze.py

```python
#!/usr/bin/env python3
"""分析实验结果"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.tsv', sep='\t')

# 按 val_psnr 排序
df = df.sort_values('val_psnr', ascending=False)

print("\n实验结果排名：")
print(df.to_string(index=False))

print(f"\n最佳实验: {df.iloc[0]['experiment']}")
print(f"最佳 PSNR: {df.iloc[0]['val_psnr']:.2f} dB")

# 绘制优化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(df)), df['val_psnr'].values, 'b-o')
plt.axhline(y=df.iloc[0]['val_psnr'], color='r', linestyle='--', label='Best')
plt.xlabel('Experiment Order')
plt.ylabel('val_psnr (dB)')
plt.title('Optimization Progress')
plt.legend()
plt.savefig('optimization_curve.png')
print("\n优化曲线已保存: optimization_curve.png")
```

## 注意事项

1. **永不停止原则**：一旦开始实验循环，不要询问"是否继续"，持续运行直到人为中断
2. **定期保存**：即使 status=discard，也保留实验目录（除非磁盘空间不足）
3. **记录充分**：description 字段要写清楚改动内容，便于后续分析
4. **避免过拟合**：如果 val_psnr 持续上升但 visual 结果变差，可能是过拟合
5. **版本控制**：experiments/ 目录可加入 .gitignore，但 results.tsv 必须提交

## 与 lv1_macro 的关系

```
lv1_macro                    lv2_micro
─────────                    ─────────
横向比较                      纵向优化
哪个方法更好？                这个方法怎么调最好？
             
03_method_edsr  ─────────►   experiments/20260412_baseline
val_psnr: 34.15              val_psnr: 34.15 (基线)
                             val_psnr: 34.72 (优化后)
                              
最终输出：优化后的 EDSR 配置
         ↓
    lv3_fusion (可选)
```

## 参考

- [autoresearch](https://github.com/karpathy/autoresearch) - 原始设计灵感
- [program.md](../program.md) - 任务目标和评估标准
