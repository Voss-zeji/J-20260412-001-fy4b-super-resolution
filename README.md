# FY-4B 卫星图像超分辨率

基于 [autoresearch](https://github.com/karpathy/autoresearch) 思想的实验性超分辨率研究项目。

## 简介

使用 PFT-SR (Progressive Feature Transfer for Super-Resolution) 方法，将 FY-4B 卫星 4km 分辨率图像提升至 2km 分辨率。

## 核心思想

参考 autoresearch 的设计原则:

| 文件 | 职责 | 修改权限 |
|------|------|----------|
| `utils.py` | 固定工具函数 (指标计算、检查点、可视化) | **AI 不可修改** |
| `main.py` | 实验代码 (模型、训练循环、超参数) | **AI 可修改** |
| `program.md` | 任务指令和目标 | **用户确认后更新** |

## 项目结构

```
.
├── main.py              # 主实验文件 (AI 修改)
├── utils.py             # 固定工具函数
├── data/                # 数据加载模块
│   └── fy4b_dataset.py
├── preprocessing/       # 数据预处理
│   └── fy4b_calibration.py
├── program.md           # 任务指令
├── progress.md          # 进度记录
├── README.md            # 本文件
├── pyproject.toml       # Python 配置
├── results/             # 实验结果输出
└── tmp/                 # 临时文件
```

## 快速开始

```bash
# 训练 CH07 通道
python main.py --band CH07

# 训练 CH08 通道
python main.py --band CH08

# 测试模式 (只验证)
python main.py --band CH07 --test

# 恢复训练
python main.py --band CH07 --checkpoint results/CH07/model_best.pth
```

## 评估指标

**核心指标**: `val_psnr` (越高越好)
- 基准: ~30 dB (bicubic)
- 目标: >35 dB

**辅助指标**: `val_ssim`, `val_loss`

## 实验流程

1. **创建分支**:
   ```bash
   git checkout -b experiment/20260412-lr-tune
   ```

2. **修改 main.py**: 调整 Config 类中的超参数或修改模型结构

3. **运行实验**:
   ```bash
   python main.py --band CH07
   ```

4. **记录结果**: 程序自动输出 `val_psnr` 等关键指标

5. **决策**:
   - 结果提升 → `git commit` 保留
   - 结果下降 → `git reset` 回退

## 数据来源

远程服务器: `gpu-server` (AutoDL)

- CH07 2km: `/root/autodl-tmp/Calibration-FY4B/2000M/CH07/`
- CH07 4km: `/root/autodl-tmp/Calibration-FY4B/4000M/CH07/`
- CH08 2km: `/root/autodl-tmp/Calibration-FY4B/2000M/CH08/`
- CH08 4km: `/root/autodl-tmp/Calibration-FY4B/4000M/CH08/`

## 环境

```bash
conda activate linpy311
pip install -e .
```

## 参考

- [PFT-SR](https://github.com/CVL-UESTC/PFT-SR) - 基础模型架构
- [autoresearch](https://github.com/karpathy/autoresearch) - 实验方法论
