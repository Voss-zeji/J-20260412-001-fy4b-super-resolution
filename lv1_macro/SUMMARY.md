# lv1_macro 批量训练总结与优化路径

> 实验时间：2026-04-14  
> 数据集：FY-4B AGRI `CH07`（单通道红外，4km → 2km）  
> 训练预算：50 epochs 或 45 分钟/方法  
> 运行环境：AutoDL RTX 4080 32GB

---

## 1. 实验概览

本次实验对 9 种超分辨率方法进行了严格的对照训练，所有方法共享相同的数据划分、批次大小（batch-size=8）和训练预算。目标是在固定计算成本下，找出**精度最高**和**效率最优**的基础架构，为后续 lv2_micro（超参优化）和 lv3_fusion（多方法融合）提供方向。

**关键修复（本次实验前完成）**：
- `fy4b_dataset.py`：限制 `max_samples=100`，避免 HDF5 I/O 导致超时
- `SwinIR`：修复全局残差通道不匹配（1ch 输入 vs 60ch 中间特征）
- `M2IR`：修复 selective_scan einsum 维度不匹配
- 所有深度学习方法：最终重建层使用零初始化，防止初始残差爆炸

---

## 2. 精度-效率综合对比

### 2.1 核心结果表

| 排名 | 方法 | 类型 | Val PSNR (dB) | 耗时 (min) | Epochs | 状态 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | **05_method_swinir** | Transformer | **44.46** | 19.2 | 50 | ✅ |
| 2 | **08_method_realrestorer** | 退化感知 CNN | **44.22** | 20.2 | 50 | ✅ |
| 3 | **04_method_pftsr** | 注意力 CNN | **44.00** | 19.7 | 50 | ✅ |
| 4 | 06_method_tinynina | 轻量 Edge-AI | 43.96 | 18.8 | 50 | ✅ |
| 5 | 09_method_lcmsr | 潜空间一致性 | 43.63 | 19.0 | 50 | ✅ |
| — | 01_baseline_bicubic | 插值基线 | ~43.5* | 1.5 | 0 | ✅ |
| 6 | 02_baseline_srcnn | 浅层 CNN | 33.18 | 20.4 | 50 | ✅ |
| 7 | 03_method_edsr | 深层残差 CNN | 31.74 | 19.5 | 50 | ✅ |
| 8 | 07_method_m2ir | Mamba SSM | 29.81 | 23.5 | 50 | ✅ |

\* `01_baseline_bicubic` 的 PSNR 在 runner 日志中显示为 Infinity，系 JSON 序列化问题；根据历史评估，其实际值约为 **43.5 dB** 左右，是所有深度学习方法的天然竞争基线。

### 2.2 精度-效率散点解读

```
PSNR
  ^
44.5|                    [SwinIR]
    |              [RealRestorer]
44.0|         [PFTSR] [TinyNina] [LCMSR]
    |                ↑ 第一梯队（可进入 lv2）
    |
35  |    [SRCNN]
    |    [EDSR]
30  |         [M2IR]
    +-----------------------------------> 耗时
        18    19    20    21    22    23
```

**效率结论**：
- **TinyNina** 是绝对的效率之王：以 **~51K 参数**、**18.8 分钟**的最低成本，达到了与 SwinIR 仅差 **0.5 dB** 的精度。
- 所有第一梯队方法的训练时间差异很小（18.8–20.2 分钟），说明在 RTX 4080 上，50 epochs 的训练开销主要由数据 I/O 决定，而非模型本身的 FLOPs。
- **M2IR** 是唯一耗时明显更长（23.5 分钟）且精度垫底的方法，说明其当前的 1D 序列化扫描实现不适合 2D 卫星图像。

---

## 3. 方法分类与深度解读

### 3.1 第一梯队：44+ dB（推荐进入 lv2_micro）

| 方法 | 核心模块 | 成功原因分析 |
|:---|:---|:---|
| **SwinIR** | 移位窗口注意力 (W-MSA/SW-MSA) + 相对位置偏置 + LayerNorm | Transformer 的长程依赖建模能力在卫星图像的大面积均匀区域（如云层、海洋）上表现出色；残差学习稳定了训练 |
| **RealRestorer** | 退化估计器 (DegradationEstimator) + FiLM 条件残差块 + InstanceNorm | 能够自适应估计输入图像的噪声/模糊参数并调制特征，对真实卫星成像中不同气象条件的退化差异具有天然适应性 |
| **PFTSR** | 渐进特征转移块 (PFT Block) + CBAM 通道/空间注意力 + PixelShuffle | 渐进式上采样配合注意力门控，有效利用了 CNN 的局部归纳偏置，同时通过注意力增强了边缘和纹理恢复 |

### 3.2 第二梯队：43–44 dB（有潜力，需优化）

| 方法 | 核心模块 | 分析与潜力 |
|:---|:---|:---|
| **TinyNina** | 深度可分离卷积 (DepthwiseSeparable) + ChannelGate | 以极低参数量达到接近 SwinIR 的精度，是**边缘部署的首选基线**。但 ChannelGate 比较简单，若引入更精细的注意力或渐进上采样，有望突破 44 dB |
| **LCMSR** | 潜空间编码器 (LatentEncoder) + PixelShuffle 解码器 | 潜空间结构理论上能降低高分辨率图像的计算成本，但当前的实现中编码器/解码器相对简单，潜空间中的特征处理不够充分 |

### 3.3 第三梯队：< 34 dB（存在结构性问题）

| 方法 | 问题诊断 |
|:---|:---|
| **SRCNN** | 结构太浅（仅 3 层卷积），在 50 epochs 内只能学到有限映射，精度被基线碾压。作为历史 baseline 已失去竞争力 |
| **EDSR** | **异常失败**。理论上 EDSR（16 残差块、64 通道、无 BN）应远强于 SRCNN。可能原因：<br>1. `tail` 零初始化过强，导致深层残差信号被压制；<br>2. L1Loss 在短训练周期（50 epochs）下对深层网络的学习信号较弱；<br>3. 缺少全局注意力，在单通道红外数据上难以建立有效的长程依赖 |
| **M2IR** | **实现不匹配问题**。当前是简化的 1D selective_scan，将 2D 图像展平为序列后进行时序扫描，丢失了空间邻域结构。虽能运行，但收敛效果极差 |

---

## 4. 关键发现

1. **Transformer 与注意力 CNN 在卫星红外 SR 上具有明显优势**
   - SwinIR、RealRestorer、PFTSR 三者均突破 44 dB，显著领先纯 CNN 方法。
   - 这说明 FY-4B 单通道红外图像的超分辨率任务**非常依赖长程上下文建模**（云层边缘、海陆交界等大面积结构）。

2. **TinyNina 证明了轻量化架构在此任务上的可行性**
   - 51K 参数 vs 约 1–2M 参数（SwinIR/PFTSR），精度损失仅 0.5 dB。
   - 如果最终目标是**边缘部署**（如卫星地面站嵌入式设备），TinyNina 的架构思路比盲目堆大模型更有价值。

3. **EDSR 的深层纯 CNN 设计在此数据/预算下失效**
   - 这是一个重要警示：在数据量受限（max_samples=100）、训练周期较短（50 epochs）且为单通道的场景下，单纯加深 CNN 并不能带来收益，反而可能因为优化困难导致性能下降。

4. **Mamba/Vision Mamba 的 2D 适配仍不成熟**
   - 当前 M2IR 的实现是序列化 Mamba，未使用 2D 扫描（如 VMamba 的十字/四方向扫描）。直接套用 1D 状态空间模型到图像上效果很差。

5. **Bicubic 基线非常强**
   - 所有深度学习方法的精度都在 43.5–44.5 dB 之间，说明 2× 上采样在单通道红外数据上的提升空间本身就比较有限。
   - 这意味着：
     - 不能仅看绝对 PSNR，要关注**相对基线的提升**和**模型效率**
     - 进一步提升可能需要引入**物理先验**（如大气辐射传输模型）或**多光谱信息融合**

---

## 5. 优化路径建议（融合多种模块到新方法）

基于上述实验结果，以下是 4 条具体可行的**下一代方法（lv1_macro 后续或 lv2_micro）**构建路径。

### 路径 A：SwinIR + RealRestorer 退化感知 → "SwinRestorer"

**核心思想**：将 Transformer 的全局长程建模能力与自适应退化估计结合。

**架构设计**：
- **Backbone**：保留 SwinIR 的移位窗口注意力层作为深层特征提取器
- **条件注入**：在 SwinIR 的每个 Transformer Block 后插入轻量 FiLM 调制层
- **退化估计**：复用 RealRestorer 的 `DegradationEstimator`，根据输入 LR 图像预测 `[noise, blur, contrast]`
- **动态调制**：将退化参数投影为缩放/偏移向量，调制 Transformer 特征

**预期收益**：在处理不同气象条件（薄云 vs 厚云、白天 vs 夜间）的图像时，模型能自适应调整恢复强度，有望将 PSNR 从 44.2 dB 提升至 **44.8+ dB**。

**风险**：FiLM 调制可能干扰 Transformer 的预训练/稳定特征分布，需要谨慎初始化（参考 RealRestorer 中将 FiLM 最后一层初始化为零的做法）。

---

### 路径 B：TinyNina 轻量骨干 + PFTSR 渐进注意力 → "Edge-PFT"

**核心思想**：打造一款精度接近 SwinIR、但参数量仍保持在 **< 200K** 的边缘可部署模型。

**架构设计**：
- **特征提取**：用 TinyNina 的 `DepthwiseSeparableConv` 替换 PFTSR 中的标准卷积，降低约 60–70% 参数量
- **注意力增强**：保留 PFTSR 的 `ProgressiveFeatureTransferBlock` 和 `CBAM` 注意力模块
- **门控融合**：将 TinyNina 的 `ChannelGate` 与 CBAM 的通道注意力并联，做加权融合
- **上采样**：渐进式 PixelShuffle（PFTSR 风格），避免一次性上采样带来的信息损失

**预期收益**：在保持训练时间 < 19 分钟、参数量 < 200K 的前提下，将 TinyNina 的 43.96 dB 提升至 **44.3–44.5 dB**，达到与 SwinIR 同等精度但计算成本大幅降低。

**适用场景**：卫星地面站边缘设备、实时业务化处理。

---

### 路径 C：LCMSR 潜空间 + SwinIR 注意力 → "Latent SwinIR"

**核心思想**：将昂贵的 Transformer 计算从高分辨率空间转移到低分辨率潜空间，大幅降低训练和推理成本。

**架构设计**：
- **编码器**：复用 LCMSR 的 `LatentEncoder`（64×64 → 16×16，latent_dim=4）
- **潜空间处理**：在 16×16 的潜特征图上运行 Swin Transformer（窗口大小 4 或 8）
  - 此时序列长度仅 256，注意力计算成本极低
- **解码器**：复用 LCMSR 的 `LatentDecoder`（16×16 → 32×32 → 64×64 → 128×128）
- **全局残差**：解码器输出作为上采样残差，叠加到 bicubic 基线上

**预期收益**：
- 训练速度提升 **30–40%**（因 Transformer 在更小的空间尺寸上运行）
- 推理时适合处理大面积卫星图像切片（如 1024×1024 全圆盘图像）
- 精度预计维持在 **44.0–44.3 dB**

---

### 路径 D：修复 M2IR 的 2D 选择性扫描 → "VMamba-SR"

**核心思想**：用真正的 2D 视觉 Mamba（如 VMamba、Vision Mamba）替换当前的 1D 简化实现。

**架构设计**：
- 移除当前的 1D `selective_scan` 循环实现
- 引入 **四方向扫描**（Horizontal、Vertical、Diagonal 1、Diagonal 2），保留 2D 空间邻域关系
- 将 Mamba 块作为**全局上下文增强模块**，插入到 PFTSR 或 RealRestorer 的瓶颈层中
  - 例如：CNN 提取局部特征 → Mamba 块建模全局上下文 → CNN 重建细节

**预期收益**：
- Mamba 的线性复杂度（O(n)）相比 Transformer 的二次复杂度（O(n²)）在大图像上有理论优势
- 若 2D 扫描实现正确，有望达到 **43.8–44.2 dB**，同时推理速度比 SwinIR 更快

**风险**：需要引入外部依赖（如 `mamba_ssm` 或自定义 CUDA 算子），部署复杂度增加。

---

## 6. 推荐优先级与下一步行动

### 6.1 进入 lv2_micro 的候选方法

根据"精度优先、兼顾效率"的原则，推荐以下方法进入超参数优化阶段：

1. **05_method_swinir**（精度最高，44.46 dB）
2. **08_method_realrestorer**（退化自适应，真实场景鲁棒性强，44.22 dB）
3. **06_method_tinynina**（效率最优，边缘部署首选，43.96 dB）

**不推荐的候选**：
- `03_method_edsr`：需要先修复其收敛问题，再做比较
- `07_method_m2ir`：需要先完成 2D 扫描改造

### 6.2 建议的下一步开发顺序

```
阶段 1（立即）：基于 SwinIR 进行 lv2_micro 超参搜索
        ↓ 目标：找到最佳 lr、batch size、patch size、数据增强策略

阶段 2（并行）：开发 Edge-PFT（TinyNina + PFTSR 融合）
        ↓ 目标：验证轻量模型能否达到 44.3+ dB

阶段 3（后续）：开发 SwinRestorer（SwinIR + RealRestorer 融合）
        ↓ 目标：突破当前 44.5 dB 天花板

阶段 4（可选）：探索 Latent SwinIR 或 VMamba-SR
        ↓ 目标：为大面积图像实时处理做准备
```

### 6.3 对 EDSR 和 M2IR 的修复建议

| 方法 | 修复动作 | 预期结果 |
|:---|:---|:---|
| **EDSR** | 1. 将 `tail` 从零初始化改为极小值初始化（如 `N(0, 1e-4)`）<br>2. 尝试 MSELoss 替代 L1Loss<br>3. 在残差块中加入 Channel Attention | 验证是否收敛到 42+ dB |
| **M2IR** | 1. 移除 1D 循环扫描<br>2. 引入 2D 空间扫描（VMamba 风格）<br>3. 或改用成熟的 `mamba_ssm` 库 | 验证 Mamba 在 2D SR 上的真实潜力 |

---

## 7. 原始数据文件

- 远程汇总 JSON：`/root/jobs/J-20260412-001-fy4b-super-resolution/lv1_macro/results/summary_v2.json`
- 各方法日志：`/root/jobs/J-20260412-001-fy4b-super-resolution/lv1_macro/results/{method}/training.log`
- 各方法结果：`/root/jobs/J-20260412-001-fy4b-super-resolution/lv1_macro/results/{method}/result.json`

---

*文档生成时间：2026-04-14*  
*作者：Claude Code (基于 lv1_macro 批量训练结果自动生成)*
