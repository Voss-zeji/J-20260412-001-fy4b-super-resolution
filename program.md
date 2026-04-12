# FY-4B 超分辨率研究

## 任务目标

使用 PFT-SR (Progressive Feature Transfer for Super-Resolution) 方法，对 FY-4B 卫星 AGRI 仪器数据进行超分辨率处理。

**核心任务**: 将 4km 分辨率图像提升至 2km 分辨率 (4000M → 2000M)

**支持通道**:
- CH07 (IR3.90, 中红外)
- CH08 (IR6.20, 中红外)

## 实验规范

### 文件职责 (关键)

| 文件 | 职责 | 修改权限 |
|------|------|----------|
| `utils.py` | 固定工具函数 (PSNR/SSIM计算、检查点管理、可视化) | **AI 不可修改** |
| `main.py` | 主实验代码 (模型、训练循环、优化器) | **AI 可修改** |
| `program.md` | 任务指令和目标定义 | **用户确认后更新** |

### 评估指标

**单一核心指标**: `val_psnr` (验证集 PSNR, 单位 dB)
- **越高越好**
- 基准线: ~30 dB (bicubic 插值)
- 目标: >35 dB

**辅助指标** (仅参考):
- `val_ssim`: 结构相似性
- `val_loss`: 验证损失

### 数据来源

**远程服务器**: gpu-server (AutoDL)
- **CH07 训练数据**:
  - 高分辨率 (2km): `/root/autodl-tmp/Calibration-FY4B/2000M/CH07/`
  - 低分辨率 (4km): `/root/autodl-tmp/Calibration-FY4B/4000M/CH07/`
- **CH08 训练数据**:
  - 高分辨率 (2km): `/root/autodl-tmp/Calibration-FY4B/2000M/CH08/`
  - 低分辨率 (4km): `/root/autodl-tmp/Calibration-FY4B/4000M/CH08/`

### 实验流程

1. **创建分支**: 每个实验在独立分支进行
   ```bash
   git checkout -b experiment/<日期>-<描述>
   ```

2. **修改 main.py**: 尝试新的实验想法
   - 可修改: 模型架构、超参数、优化器、损失函数
   - 不可修改: utils.py 中的工具函数

3. **运行实验**:
   ```bash
   # CH07 通道
   python main.py --band CH07

   # CH08 通道
   python main.py --band CH08
   ```

4. **记录结果**: 自动输出关键指标
   ```
   val_psnr:     32.45
   val_ssim:     0.8923
   train_time:   300.5s
   ```

5. **决策**:
   - **val_psnr 提升** → 保留提交 (`git commit`)
   - **val_psnr 下降** → 回退 (`git reset`)
   - **崩溃/异常** → 修复或放弃

### 实验参数

**固定参数** (不建议修改):
- 输入/输出通道: 1 (单通道训练)
- 上采样因子: 2x
- 图像块大小: 64 (低分辨率空间)
- 数据归一化范围: [150K, 350K] → [-1, 1]

**可调参数** (在 main.py 中修改):
- 模型深度 (`num_pft_blocks`)
- 特征维度 (`num_features`)
- 学习率、优化器类型
- 损失函数权重
- 批次大小
- 训练 epoch 数

### 简化准则

- **简单优于复杂**: 0.1 dB 提升但增加 50 行代码 → 不值得
- **删除优于添加**: 删除代码获得相同效果 → 值得保留
- **单一文件**: 所有实验修改集中在 main.py

## 待办事项

- [x] 调研 FY-4B 卫星数据特点
- [x] 调研 PFT-SR 超分辨率算法
- [x] 搭建基础训练框架
- [x] 实现 CH07 通道训练
- [ ] 优化模型达到 >35 dB PSNR
- [ ] 实现 CH08 通道训练
- [ ] 对比实验 (不同架构、损失函数)

## 实验记录

| 日期 | 分支 | val_psnr | 改动描述 | 状态 |
|------|------|----------|----------|------|
| 2026-04-12 | baseline | ~30.0 | Bicubic 基准 | 基准 |
| 2026-04-12 | main | ~32.5 | PFT-SR 基础实现 | 当前 |

## 参考

- PFT-SR 论文: https://github.com/CVL-UESTC/PFT-SR
- FY-4B 数据说明: preprocessing/README_FY4B_Calibration.md
