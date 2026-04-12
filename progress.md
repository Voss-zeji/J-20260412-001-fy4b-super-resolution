# 进度记录 - FY-4B 超分辨率研究

## 任务信息
- **任务ID**: J-20260412-001
- **创建日期**: 2026-04-12
- **状态**: 进行中
- **方法论**: 三层实验架构（参考 autoresearch）

## 执行日志

### 2026-04-12
- [x] 创建任务目录结构
- [x] 从 GPU 服务器复制 FY4B 超分辨率代码
- [x] 按照本地规范重新配置文件
- [x] **设计三层实验架构**
  - 创建 `ARCHITECTURE.md` - 架构设计文档
  - **lv1_macro** (宏观层): 方法间比较
    - 创建 `lv1_macro/README.md` - 统一接口规范
    - 设计方法目录结构：`methods/XX_category_name/`
    - 定义公平比较原则：相同数据、预算、指标
  - **lv2_micro** (微观层): 模型内优化（autoresearch 模式）
    - 创建 `lv2_micro/README.md` - autoresearch 工作流
    - 创建 `run_experiment.sh` - 自动化实验脚本
    - 创建 `analyze.py` - 结果分析工具
    - 设计实验记录格式：`results.tsv`
  - **lv3_fusion** (融合层): 方法组合创新
    - 创建 `lv3_fusion/README.md` - 融合策略说明
    - 定义触发条件和决策标准
- [x] 更新 `program.md` - 完整的三层架构说明
- [x] 更新 `.gitignore` - 添加输出目录忽略规则
- [x] 初始提交到 git (commit: 323bccb)

## 待办清单

### lv1_macro - 宏观层
- [ ] 收集候选方法代码
  - [ ] 01_baseline_bicubic (双三次插值)
  - [ ] 02_baseline_srcnn (SRCNN)
  - [ ] 03_method_edsr (EDSR)
  - [ ] 04_method_pftsr (PFT-SR，已有)
  - [ ] 05_method_swinir (SwinIR)
- [ ] 实现统一接口适配
- [ ] 运行宏观比较 (CH07)
- [ ] 选择最佳方法进入 lv2_micro

### lv2_micro - 微观层
- [ ] 初始化基线实验
- [ ] 运行 autoresearch 优化循环
- [ ] 达到目标 PSNR > 35 dB

### lv3_fusion - 融合层（可选）
- [ ] 检查触发条件（top-2 gap < 0.5 dB）
- [ ] 实现融合策略（如需要）

### 验证与输出
- [ ] 在 CH08 通道验证泛化性
- [ ] 生成对比可视化
- [ ] 保存最终模型

## 实验记录

### lv1_macro - 宏观比较

| 日期 | 方法 | val_psnr | val_ssim | 参数量 | 状态 |
|------|------|----------|----------|--------|------|
| - | 01_baseline_bicubic | - | - | 0 | 待运行 |
| - | 02_baseline_srcnn | - | - | - | 待运行 |
| - | 03_method_edsr | - | - | - | 待运行 |
| - | 04_method_pftsr | - | - | 2.8M | 待运行 |
| - | 05_method_swinir | - | - | - | 待运行 |

### lv2_micro - 微观优化

| 日期 | 实验 | val_psnr | 改动 | 状态 |
|------|------|----------|------|------|
| - | - | - | - | 待开始 |

## 项目结构（三层架构）

```
.
├── ARCHITECTURE.md              # 架构设计文档（核心）
├── program.md                   # 任务指令
├── progress.md                  # 进度记录
├── compare.py                   # 统一比较入口
├── utils.py                     # 固定工具（不可修改）
├── data/                        # 统一数据加载
│   └── fy4b_dataset.py
├── lv1_macro/                   # ========== 宏观层 ==========
│   ├── README.md                # 宏观层说明
│   ├── methods/                 # 各方法独立目录
│   │   ├── 01_baseline_bicubic/
│   │   ├── 02_baseline_srcnn/
│   │   ├── 03_method_edsr/
│   │   ├── 04_method_pftsr/
│   │   └── 05_method_swinir/
│   ├── results.csv              # 比较结果表
│   └── select_best.py           # 选择最佳方法
├── lv2_micro/                   # ========== 微观层 ==========
│   ├── README.md                # 微观层说明（autoresearch）
│   ├── TARGET_METHOD            # 当前优化目标
│   ├── experiments/             # 实验目录
│   ├── results.tsv              # 实验记录
│   ├── run_experiment.sh        # 自动化脚本
│   └── analyze.py               # 结果分析
├── lv3_fusion/                  # ========== 融合层 ==========
│   └── README.md
├── preprocessing/               # 数据预处理
└── results/                     # 最终输出
```

## 三层架构与 autoresearch 的对应

| autoresearch | 本架构对应 | 说明 |
|--------------|-----------|------|
| `train.py` | `lv2_micro/experiments/*/main.py` | 单一文件迭代 |
| `prepare.py` | `utils.py` + `data/` | 固定工具 |
| `program.md` | `program.md` + `ARCHITECTURE.md` | 任务定义 |
| `val_bpb` | `val_psnr` | 单一核心指标 |
| git commit/reset | `lv2_micro` 层 | 微观决策 |
| 永不停止 | `run_experiment.sh` | 自动化循环 |

## 使用方法速查

```bash
# lv1_macro: 宏观比较
python compare.py --level macro --band CH07

# lv2_micro: 微观优化（autoresearch模式）
cd lv2_micro
./run_experiment.sh CH07 20260412_baseline "baseline"

# 分析结果
python analyze.py --plot --suggest

# lv3_fusion: 融合（可选）
cd ../lv3_fusion
python fusion.py --methods edsr,pftsr
```

## 参考

- [ARCHITECTURE.md](ARCHITECTURE.md) - 详细架构设计
- [autoresearch](https://github.com/karpathy/autoresearch) - 原始灵感
- [PFT-SR](https://github.com/CVL-UESTC/PFT-SR) - 基础模型
