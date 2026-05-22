# FY-4B 超分辨率 LV1+LV2 综合分析与后续训练计划

**整理日期**: 2026-05-22
**实验波段**: CH07 (3.9μm 红外通道)
**硬件**: NVIDIA RTX 4080 SUPER (32GB VRAM)

---

## 一、全量结果总表（LV1 + LV2，20 种方法）

按 PSNR 降序排列。LV2 论文方法标注 12ep 估计值。

| 排名 | 层级 | 方法 | 架构 | PSNR | SSIM | 参数量 | 推理(ms) | 大小 | Epochs |
|:----:|:----:|-------|------|:----:|:----:|-------:|:--------:|-----:|:------:|
| 1 | LV1 | **PFTSR** | CNN+CBAM+渐进转移 | **44.74** | **0.9831** | 1.00M | 52.1 | 3.8MB | 50 |
| 2 | LV1 | **RealRestorer** | CNN+退化感知FiLM | 44.69 | 0.9811 | 1.04M | 19.7 | 4.0MB | 50 |
| 3 | LV1 | **TinyNina** | 轻量CNN+通道门控 | 44.39 | 0.9808 | **45K** | **10.1** | **0.2MB** | 50 |
| 4 | LV1 | SwinIR | Transformer+移位窗口 | 44.11 | 0.9822 | 406K | 13.3 | 1.6MB | 50 |
| 5 | LV2 | SwinRestorer | SwinIR+退化感知FiLM | 44.03 | 0.9809 | 581K | 36.6 | 2.2MB | 50 |
| 6 | LV2 | DualScaleRestorer | EDSR+TinyNina+退化融合 | 43.80 | 0.9781 | 1.59M | 34.2 | 6.1MB | 50 |
| 7 | LV2 | Edge-PFT | TinyNina轻量+PFTSR渐进 | 43.73 | 0.9760 | 94K | 20.3 | 0.4MB | 50 |
| 8 | LV1 | LCMSR | 潜空间一致性扩散 | 43.77 | 0.9768 | 2.75M | 17.6 | 10.5MB | 50 |
| 9 | LV2 | EmambaIR | Mamba SSM+全局建模 | 43.67 | 0.9795 | — | — | — | 12* |
| 10 | LV2 | MambaPFT | Mamba+PFTSR混合 | 43.49 | 0.9789 | 1.22M | 66.5 | 4.6MB | 50 |
| 11 | LV2 | LatentSwinIR | 潜空间+SwinIR | 43.36 | 0.9765 | 5.06M | 40.7 | 19.3MB | 50 |
| 12 | LV1 | EDSR | 深层残差CNN | 42.79 | 0.9810 | 1.37M | 20.5 | 5.2MB | 50 |
| 13 | LV2 | NTIRE2026 IR SR | 遥感红外SR | 42.85 | 0.9755 | — | — | — | 12* |
| 14 | LV2 | GPROF-IR | 红外降水增强 | 42.67 | 0.9737 | — | — | — | 12* |
| 15 | LV2 | IMPA-Net | 气象多尺度注意力 | 42.40 | 0.9745 | — | — | — | 12* |
| 16 | LV1 | SRCNN | 3层CNN | 40.84 | 0.9717 | 8K | 14.6 | 0.03MB | 50 |
| 17 | LV1 | M2IR | Mamba SSM | 33.42 | — | — | — | — | 20† |
| 18 | LV2 | Weather SR | 气象预报SR | 32.80 | 0.8416 | — | — | — | 11* |
| 19 | LV2 | MultiSpectral SR | 自编码器+残差CNN | 29.94 | 0.8189 | — | — | — | 12* |

\* 30min 超时，未跑满 50 epoch
† LV1 中超时，仅 20 epoch

---

## 二、核心分析

### 2.1 精度天花板与瓶颈

所有方法的 PSNR 集中在 **43-45 dB** 区间（排除训练不足和不适配的），表明：

- **43 dB 是 50 epoch 基础训练的稳定收敛区间**，继续训练到 200 epoch 预计可提升 0.5-1.0 dB
- **LV1 最佳 PFTSR (44.74 dB)** 仍然是最高，LV2 融合方法未实现突破
- **当前 PSNR 水平已远超目标 35 dB**，但与 Bicubic 基线 (~43.5 dB) 的差距不大，说明任务难度相对可控

### 2.2 架构效能分析

按精度/参数效率分三个阵营：

**高效阵营（PSNR>44, <1.1M 参数）**：
- PFTSR: 44.74 dB / 1.00M — 精度最优
- RealRestorer: 44.69 dB / 1.04M — 退化感知+推理快
- SwinIR: 44.11 dB / 406K — 精度/参数比最高
- TinyNina: 44.39 dB / 45K — 极致轻量

**中效阵营（PSNR 43-44, 各种参数量）**：
- SwinRestorer: 44.03 dB / 581K — 退化感知有价值
- Edge-PFT: 43.73 dB / 94K — 部署首选
- MambaPFT: 43.49 dB / 1.22M — 推理慢(66ms)是短板
- LatentSwinIR: 43.36 dB / 5.06M — 参数最大，性价比最低

**低效/不适配（PSNR < 34）**：
- Weather SR, MultiSpectral SR, M2IR — 架构不适配或训练不足

### 2.3 关键结论

1. **PFTSR 的渐进特征转移 + CBAM 注意力**是最适合 FY-4B 亮温 SR 的架构范式
2. **退化感知 (FiLM) 有效但引入开销**：RealRestorer 精度接近 PFTSR，SwinRestorer 加入 FiLM 后推理从 13ms 增至 37ms
3. **Mamba SSM 在此任务上不如 CNN/Transformer**：M2IR 训练困难，MambaPFT 推理最慢
4. **论文"相关性"≠ 实际效果**：Weather SR 论文与气象极相关但表现最差，原因是数据域偏移
5. **轻量化可行**：TinyNina (45K) 和 Edge-PFT (94K) 均达到 43.7+ dB，证明轻量部署不是问题

---

## 三、淘汰与保留

### 淘汰（5 个）

| 方法 | 淘汰原因 |
|------|----------|
| M2IR | 训练不稳定，33.42 dB (20ep)，Mamba 2D 扫描实现有根本问题 |
| Weather SR | 32.80 dB，气象预报架构不适配亮温 SR |
| MultiSpectral SR | 29.94 dB，自编码器架构不适配 |
| LatentSwinIR | 43.36 dB + 5.06M 参数，参数效率最差，推理无优势 |
| MambaPFT | 43.49 dB + 66.5ms 推理，速度最慢，不如直接用 CNN |

### 保留进入后续训练（8 个）

| 方法 | 保留理由 | 当前PSNR | 预期潜力 |
|------|----------|:--------:|----------|
| PFTSR | 精度最高，核心架构 | 44.74 | 45.0+ (200ep) |
| RealRestorer | 退化感知+快速推理 | 44.69 | 45.0+ (200ep) |
| TinyNina | 极致轻量，部署首选 | 44.39 | 44.8+ (200ep) |
| SwinIR | Transformer 精度骨干 | 44.11 | 44.5+ (200ep) |
| SwinRestorer | 精度+退化感知 | 44.03 | 44.5+ (200ep) |
| DualScaleRestorer | 双尺度互补设计 | 43.80 | 44.3+ (200ep) |
| Edge-PFT | 轻量部署最优解 | 43.73 | 44.2+ (200ep) |
| EmambaIR | 论文方法最佳，趋势好 | 43.67* | 44.0+ (50ep) |

\* 仅 12 epoch，完整训练预期大幅提升

---

## 四、后续训练计划

### Phase 1：补全训练（优先级最高）

**目标**：EmambaIR + NTIRE2026 + GPROF-IR + IMPA-Net 跑满 50 epoch

| 方法 | 已跑 | 待跑 | 预计时间 | 理由 |
|------|:----:|:----:|----------|------|
| EmambaIR | 12ep | 50ep | ~2h | 12ep 即 43.67，趋势好 |
| NTIRE2026 IR SR | 12ep | 50ep | ~2h | 遥感红外直接相关 |
| GPROF-IR | 12ep | 50ep | ~2h | 红外通道增强 |
| IMPA-Net | 12ep | 50ep | ~2h | 气象多尺度注意力 |

**执行方式**：
```bash
# GPU 服务器上运行，每个方法单独跑 50 epoch，无超时限制
cd ~/jobs/J-20260412-001-fy4b-super-resolution/lv2_micro/lv2-save
python run_all.py --band CH07 --max-epochs 50 --timeout 120 --methods 17 15 18 19
```

**预计总时间**：约 8 小时

### Phase 2：深度训练（核心方法 200 epoch）

**目标**：Top-4 方法延长至 200 epoch，观察收敛上限

| 方法 | 当前 (50ep) | 目标 | 学习率策略 | 预计时间 |
|------|:-----------:|:----:|------------|----------|
| PFTSR | 44.74 | 200ep | CosineAnnealing 1e-4→1e-6 | ~4h |
| RealRestorer | 44.69 | 200ep | CosineAnnealing 1e-4→1e-6 | ~4h |
| TinyNina | 44.39 | 200ep | CosineAnnealing 1e-4→1e-6 | ~3.5h |
| SwinRestorer | 44.03 | 200ep | CosineAnnealing 1e-4→1e-6 | ~4h |

**预计总时间**：约 16 小时（可分批跑）

### Phase 3：CH08 通道泛化验证

**目标**：在 CH08 (10.8μm) 通道验证 Top-4 的泛化能力

| 方法 | CH07 PSNR | CH08 PSNR (待测) | 差值期望 |
|------|:---------:|:----------------:|:--------:|
| PFTSR | 44.74 | ? | <1.0 dB |
| RealRestorer | 44.69 | ? | <1.0 dB |
| TinyNina | 44.39 | ? | <1.0 dB |
| SwinRestorer | 44.03 | ? | <1.0 dB |

**预计总时间**：约 5 小时（4 方法 × 50ep × ~1.2h）

### Phase 4：LV3 融合探索（可选，视 Phase 2 结果决定）

**触发条件**：Phase 2 完成后，Top-2 差值 < 0.5 dB 时启动

候选融合策略：
1. **PFTSR + RealRestorer**：渐进特征转移 + 退化感知，取两者之长
2. **TinyNina + SwinIR**：轻量局部 + Transformer 全局，双路径
3. **Ensemble**：Top-3 加权平均，零额外训练成本

---

## 五、执行时间线

| 阶段 | 内容 | 预计时长 | 开始时间 |
|------|------|----------|----------|
| Phase 1 | 补全 4 方法 50ep | 8h | 服务器空闲时 |
| Phase 2 | Top-4 深度训练 200ep | 16h | Phase 1 后 |
| Phase 3 | CH08 泛化验证 | 5h | Phase 2 后 |
| Phase 4 | LV3 融合（可选） | 8h | Phase 3 后 |

**总计约 37 小时 GPU 时间**，建议分 3-4 天完成。

---

## 六、最终建议

1. **精度优先选 PFTSR**，部署优先选 TinyNina，均衡选 RealRestorer
2. **Phase 1 必须先做**：EmambaIR 补全 50ep 后可能进入 Top-5，影响后续决策
3. **200ep 深度训练值得投入**：50ep 时 PSNR 曲线仍在上升，未出现明显平台
4. **暂时不进入 LV3 融合**：等 Phase 1-2 完成后看最终排名差距再决定

---

*本文件由 Claude Code 整理，基于 LV1 (2026-04-25) + LV2 (2026-05-20) 实验结果*
