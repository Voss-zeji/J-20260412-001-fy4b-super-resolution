# lv3_fusion - 融合层：方法组合创新

## 目标

当 lv1_macro 中多个方法表现接近时，通过融合策略组合它们的优点，获得比单一方法更好的性能。

## 触发条件

```
lv1_macro 结果示例：
┌─────────────┬──────────┬────────┐
│ Method      │ val_psnr │ Gap    │
├─────────────┼──────────┼────────┤
│ edsr        │ 34.15    │ 0.00   │ ← best
│ pftsr       │ 33.82    │ 0.33   │ ← gap < 0.5
│ swinir      │ 33.65    │ 0.50   │ ← gap < 0.5
└─────────────┴──────────┴────────┘

触发条件：top-2 或 top-3 方法的 gap < 0.5 dB
说明：这些方法各有优势，值得尝试融合
```

## 融合策略

### 策略1：输出级融合（最简单）

```python
class OutputFusion(nn.Module):
    """输出像素级加权融合"""
    def __init__(self, methods=['edsr', 'pftsr']):
        super().__init__()
        self.models = nn.ModuleList([load_model(m) for m in methods])
        # 可学习的融合权重
        self.weights = nn.Parameter(torch.ones(len(methods)) / len(methods))

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, outputs))
        return fused
```

**优点**：
- 简单，无需重新训练底层模型
- 快速验证融合是否有收益

**缺点**：
- 推理时需要同时运行多个模型
- 只能做线性组合

### 策略2：特征级融合

```python
class FeatureFusion(nn.Module):
    """中间特征融合"""
    def __init__(self):
        super().__init__()
        # 加载预训练模型
        self.encoder_edsr = load_edsr_encoder()
        self.encoder_pftsr = load_pftsr_encoder()

        # 融合层
        self.fusion_conv = nn.Conv2d(
            edsr_channels + pftsr_channels,
            fused_channels,
            kernel_size=1
        )

        # 重建层
        self.reconstruction = nn.Sequential(...)

    def forward(self, x):
        feat_edsr = self.encoder_edsr(x)
        feat_pftsr = self.encoder_pftsr(x)

        # 特征拼接 + 融合
        feat_cat = torch.cat([feat_edsr, feat_pftsr], dim=1)
        feat_fused = self.fusion_conv(feat_cat)

        return self.reconstruction(feat_fused)
```

**优点**：
- 可以学习更复杂的特征交互
- 潜在收益更大

**缺点**：
- 需要重新训练
- 设计和调优更复杂

### 策略3：多尺度融合

```python
class MultiScaleFusion(nn.Module):
    """多尺度特征融合"""
    def __init__(self):
        super().__init__()
        # 不同方法处理不同尺度
        self.model_2x = load_model('srcnn')  # 轻量级，处理细节
        self.model_4x = load_model('edsr')   # 重量级，处理全局

    def forward(self, x):
        # 多尺度处理
        sr_2x = self.model_2x(x)
        sr_4x = self.model_4x(x)

        # 高频细节 + 低频全局
        detail = sr_2x - F.avg_pool2d(sr_2x, 3, padding=1)
        base = F.avg_pool2d(sr_4x, 3, padding=1)

        return base + detail
```

## 目录结构

```
lv3_fusion/
├── README.md                    # 本文件
├── strategies/                  # 不同融合策略
│   ├── 01_output_fusion/        # 输出级融合
│   │   ├── main.py
│   │   └── train.py
│   ├── 02_feature_fusion/       # 特征级融合
│   │   ├── main.py
│   │   └── train.py
│   └── 03_multi_scale/          # 多尺度融合
│       ├── main.py
│       └── train.py
├── results.csv                  # 融合结果
└── compare_with_single.py       # 与单一方法对比
```

## 工作流程

```
阶段1：检查触发条件
├── 读取 lv1_macro/results.csv
├── 计算 top-3 方法的 gap
└── 如果 max_gap < 0.5 dB → 进入阶段2

阶段2：选择融合策略
├── 快速验证：策略1（输出级融合）
│   └── 如果提升 > 0.3 dB → 保留
└── 深度优化：策略2（特征级融合）
    └── 重新训练融合层

阶段3：验证与输出
├── 在 CH08 上验证泛化性
├── 生成融合 vs 单一方法的对比图
└── 输出最终模型到 results/final_model/
```

## 决策标准

```
融合模型 vs 最佳单一方法：

+0.5 dB 以上 → 值得保留（显著改善）
+0.3~0.5 dB → 可保留（边际改善）
+0.0~0.3 dB → 不值得（复杂度增加）
负数        → 丢弃（失败）
```

## 注意事项

1. **融合不是万能的**：如果单一方法已经有 38+ dB，融合提升空间很小
2. **考虑推理成本**：融合模型通常更慢，需要权衡精度-速度
3. **避免过拟合**：融合层参数量要小，防止过拟合到验证集
4. **可解释性**：记录哪种融合策略最有效，为后续研究提供insights

## 与 lv1/lv2 的关系

```
lv1_macro          lv2_micro              lv3_fusion
─────────          ─────────              ──────────
横向比较            纵向优化                组合创新
选最佳方法          调最优参数              融合多个方法

edsr 34.15  ─────►  34.72 (优化后)  ─────►  ?
pftsr 33.82 ─────►  -               ─────►  fusion model
                              
期望输出: fusion > max(edsr, pftsr) + 0.3 dB
```

## 参考

- Model Ensemble Techniques
- Knowledge Distillation（可作为轻量级融合替代）
- Multi-Model Fusion in Super-Resolution
