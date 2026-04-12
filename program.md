# FY-4B 超分辨率研究

## 任务目标

对 FY-4B 卫星 AGRI 仪器数据进行超分辨率处理，将 4km 分辨率图像提升至 2km 分辨率。

**核心任务**: 4000M → 2000M (4km → 2km) 超分辨率

**支持通道**:
- CH07 (IR3.90, 中红外)
- CH08 (IR6.20, 中红外)

## 核心设计：三层实验架构

本项目采用分层实验设计，结合 autoresearch 的单一指标决策思想：

```
┌─────────────────────────────────────────────────────────────────┐
│                        三层实验架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   lv1_macro (横向)        lv2_micro (纵向)       lv3_fusion    │
│   ===========             ============           ===========   │
│   方法间比较        →     方法内优化      →      方法融合      │
│   哪个方法更好？          怎么调最好？           组合更好？      │
│                                                                 │
│   ┌────────────┐                                     ┌──────┐  │
│   │ 01_bicubic │ 30.2 dB                            │fusion│  │
│   ├────────────┤          ┌──────────┐              │35.0  │  │
│   │ 02_srcnn   │ 32.5 dB  │ lr=0.001 │ 34.7 dB      └──────┘  │
│   ├────────────┤  ──────► ├──────────┤  ──────►              │
│   │ 03_edsr    │ 34.1 dB  │ depth=4  │ 34.9 dB                 │
│   ├────────────┤  ──────► └──────────┘                         │
│   │ 04_pftsr   │ 33.8 dB                                      │
│   └────────────┘          进入lv2后                            │
│        ↑                  autoresearch模式                      │
│   选择最佳方法            永不停止迭代                          │
│                                                                 │
│   决策指标: val_psnr      决策指标: val_psnr     决策: 是否>单体 │
│   动作: 选择进入lv2       动作: commit/reset     动作: 保留/放弃 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 与 autoresearch 的对应

| autoresearch | 本架构对应 | 说明 |
|--------------|-----------|------|
| `train.py` | `lv2_micro/experiments/*/main.py` | 单一文件迭代 |
| `prepare.py` | `utils.py` + `data/` | 固定工具 |
| `program.md` | `program.md` + `ARCHITECTURE.md` | 任务定义 |
| `val_bpb` 指标 | `val_psnr` | 单一核心指标 |
| git commit/reset | `lv2_micro` 层使用 | 微观决策 |
| 永不停止 | `run_experiment.sh` | 自动化循环 |

## 三层详解

### lv1_macro - 宏观层

**目标**：横向比较不同超分辨率方法，找出最适合 FY4B 数据的架构。

**目录**：`lv1_macro/methods/`

**方法列表**（编号优先级）：
```
01_baseline_bicubic/    # 双三次插值基线
02_baseline_srcnn/      # SRCNN (2015)
03_method_edsr/         # EDSR (2017)
04_method_pftsr/        # PFT-SR (自定义)
05_method_swinir/       # SwinIR (2021)
```

**使用**：
```bash
# 运行所有方法比较
python compare.py --level macro --band CH07

# 输出: lv1_macro/results.csv
# 自动选择最佳方法进入 lv2_micro
python lv1_macro/select_best.py
```

**核心原则**：
- 相同数据、相同预算（100 epoch）
- 不修改方法内部，只调用统一接口
- 单一指标 `val_psnr` 决策

### lv2_micro - 微观层

**目标**：在选定方法基础上，通过 autoresearch 模式找到最优超参数。

**目录**：`lv2_micro/experiments/`

**工作方式**（autoresearch 模式）：
```bash
cd lv2_micro

# 初始化基线
./run_experiment.sh CH07 20260412_baseline "edsr baseline"

# 迭代优化（永不停止）
while true; do
    # 修改 experiments/<name>/main.py
    ./run_experiment.sh CH07 <new_experiment> <description>
    # 自动记录到 results.tsv
    # 好则保留，差则丢弃
done
```

**可调参数**：
- 学习率 (LR): 1e-5 ~ 1e-3
- 模型深度 (NUM_PFT_BLOCKS): 2 ~ 5
- 特征维度 (NUM_FEATURES): 32, 64, 128
- 损失权重 (LAMBDA_L1, LAMBDA_SSIM)
- 批次大小 (BATCH_SIZE): 4, 8, 16

**核心原则**：
- 完全遵循 autoresearch 工作流
- 单一文件 (`main.py`) 修改
- git commit/reset 决策
- 永不停止直到人为中断

### lv3_fusion - 融合层（可选）

**目标**：当 lv1 中多个方法表现接近时，尝试融合获得更好性能。

**触发条件**：top-2 方法 gap < 0.5 dB

**策略**：
- 输出级融合（快速验证）
- 特征级融合（深度优化）
- 多尺度融合

**决策**：融合模型 > 最佳单一方法 + 0.3 dB 才保留

## 文件职责总览

| 文件/目录 | 层级 | 权限 | 说明 |
|-----------|------|------|------|
| `ARCHITECTURE.md` | 全局 | 参考 | 架构设计文档 |
| `utils.py` | 全局 | **不可修改** | 固定工具函数 |
| `data/` | 全局 | **不可修改** | 统一数据加载 |
| `lv1_macro/methods/` | lv1 | 可添加 | 各方法实现 |
| `lv1_macro/results.csv` | lv1 | 自动生成 | 比较结果 |
| `lv2_micro/TARGET_METHOD` | lv2 | 自动写入 | 当前优化目标 |
| `lv2_micro/experiments/` | lv2 | **可修改** | 实验目录（核心） |
| `lv2_micro/results.tsv` | lv2 | 自动生成 | 实验记录 |
| `lv3_fusion/` | lv3 | 可选 | 融合策略 |

## 评估指标

**单一核心指标**: `val_psnr` (验证集 PSNR, 单位 dB)
- **越高越好**
- **所有层级统一使用此指标**
- 基准线: ~30 dB (bicubic)
- 目标: >35 dB

**辅助指标** (仅参考):
- `val_ssim`: 结构相似性
- `model_params`: 模型参数量
- `train_time`: 训练时间

## 数据来源

**远程服务器**: gpu-server (AutoDL)

| 通道 | 高分辨率 (2km) | 低分辨率 (4km) |
|------|----------------|----------------|
| CH07 | `/root/autodl-tmp/Calibration-FY4B/2000M/CH07/` | `/root/autodl-tmp/Calibration-FY4B/4000M/CH07/` |
| CH08 | `/root/autodl-tmp/Calibration-FY4B/2000M/CH08/` | `/root/autodl-tmp/Calibration-FY4B/4000M/CH08/` |

## 完整工作流程

```bash
# 阶段1：宏观比较（1-2天）
python compare.py --level macro --band CH07
# → 选择最佳方法（如 03_edsr）

# 阶段2：微观优化（3-5天，autoresearch模式）
cd lv2_micro
./run_experiment.sh CH07 20260412_baseline "baseline"
# → 持续迭代，记录到 results.tsv
# → 直到收敛或人为停止

# 阶段3：融合创新（可选，1-2天）
cd ../lv3_fusion
# 如果触发条件满足
python fusion.py --methods edsr,pftsr

# 阶段4：验证与输出
python compare.py --all --band CH08  # 在CH08验证
python visualize.py --all            # 生成对比图
```

## 待办事项

- [x] 调研 FY-4B 卫星数据特点
- [x] 调研超分辨率算法
- [x] 搭建基础训练框架
- [x] 实现 PFT-SR 基础模型
- [ ] **lv1_macro**: 收集并适配所有候选方法
- [ ] **lv1_macro**: 运行宏观比较，选择最佳方法
- [ ] **lv2_micro**: 进入 autoresearch 优化模式
- [ ] **lv2_micro**: 达到 >35 dB PSNR
- [ ] **lv3_fusion** (可选): 尝试方法融合
- [ ] 在 CH08 通道验证泛化性

## 实验记录

| 日期 | 层级 | 实验 | val_psnr | 说明 |
|------|------|------|----------|------|
| 2026-04-12 | lv1 | 01_bicubic | 30.2 | 基线 |
| 2026-04-12 | lv1 | 02_srcnn | - | 待运行 |
| 2026-04-12 | lv1 | 03_edsr | - | 待运行 |
| 2026-04-12 | lv1 | 04_pftsr | 33.8 | 当前最佳候选 |

## 参考

- [ARCHITECTURE.md](ARCHITECTURE.md) - 详细架构设计
- [lv1_macro/README.md](lv1_macro/README.md) - 宏观层说明
- [lv2_micro/README.md](lv2_micro/README.md) - 微观层说明（autoresearch）
- [autoresearch](https://github.com/karpathy/autoresearch) - 原始灵感
- [PFT-SR](https://github.com/CVL-UESTC/PFT-SR) - 基础模型
