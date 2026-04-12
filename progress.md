# 进度记录 - FY-4B 超分辨率研究

## 任务信息
- **任务ID**: J-20260412-001
- **创建日期**: 2026-04-12
- **状态**: 进行中
- **方法论**: 参考 autoresearch 实验思想

## 执行日志

### 2026-04-12
- [x] 创建任务目录结构
- [x] 从 GPU 服务器复制 FY4B 超分辨率代码
- [x] 按照本地规范重新配置文件
- [x] **参考 autoresearch 优化目录结构**
  - 明确文件职责分离: `main.py` (AI修改) vs `utils.py` (固定工具)
  - 简化配置文件，将配置集中到 main.py 的 Config 类
  - 移除冗余的 models/, configs/, train.py, test.py
  - 更新 program.md 为实验指令格式
  - 更新 README.md 说明新的实验流程

## 待办清单

- [x] 调研 FY-4B 卫星数据特点（AGRI 仪器）
- [x] 收集和预处理 FY-4B 数据
- [x] 调研超分辨率算法（PFT-SR）
- [x] 建立训练数据集（低分辨率-高分辨率图像对）
- [x] 实现超分辨率模型
- [ ] 模型训练与评估（在远程GPU服务器执行）
- [ ] 结果分析与可视化

## 实验记录

| 日期 | 分支 | val_psnr | 改动描述 | 状态 |
|------|------|----------|----------|------|
| 2026-04-12 | baseline | ~30.0 | Bicubic 基准 | 基准 |
| 2026-04-12 | main | ~32.5 | PFT-SR 基础实现 | 当前 |

## 项目结构 (autoresearch 风格)

```
.
├── main.py              # 主实验文件 (AI可修改)
│   └── Config 类        # 超参数配置
│   └── PFTSR 模型       # 模型定义
│   └── 训练循环         # 训练/验证逻辑
├── utils.py             # 固定工具函数
│   └── PSNR/SSIM 计算   # (AI不可修改)
│   └── 检查点管理       # (AI不可修改)
│   └── 可视化工具       # (AI不可修改)
├── data/                # 数据加载
│   └── fy4b_dataset.py
├── preprocessing/       # 数据预处理
│   └── fy4b_calibration.py
├── program.md           # 实验指令
├── progress.md          # 进度记录
├── README.md            # 项目说明
├── pyproject.toml       # Python 配置
├── results/             # 实验结果
└── tmp/                 # 临时文件
```

## 数据来源

**远程服务器**: gpu-server (AutoDL)

| 通道 | 高分辨率 (2km) | 低分辨率 (4km) |
|------|----------------|----------------|
| CH07 | `/root/autodl-tmp/Calibration-FY4B/2000M/CH07/` | `/root/autodl-tmp/Calibration-FY4B/4000M/CH07/` |
| CH08 | `/root/autodl-tmp/Calibration-FY4B/2000M/CH08/` | `/root/autodl-tmp/Calibration-FY4B/4000M/CH08/` |

## 参考

- [autoresearch](https://github.com/karpathy/autoresearch) - 实验方法论
- [PFT-SR](https://github.com/CVL-UESTC/PFT-SR) - 模型架构
