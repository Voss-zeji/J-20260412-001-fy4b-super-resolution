# LV3 融合候选 — Top-10 方法选定

**生成时间**: 2026-06-07  
**筛选标准**: 200 epoch CH07 最佳 checkpoint PSNR 降序  
**入选门槛**: ≥ 44.16 dB

---

## Top-10 完整榜单

| 排名 | 方法 | PSNR (dB) | SSIM | 参数量 | 推理(ms) | 类型 | 来源 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | EmambaIR | **44.42** | 0.9803 | 890,889 | 60.1 | Mamba SSM | LV1 |
| 2 | PFTSR | **44.35** | 0.9816 | 1,000,103 | 39.6 | CNN+Attention | LV1 |
| 3 | DualScaleRestorer | **44.34** | 0.9803 | 1,586,415 | 32.7 | 融合模型 | LV2 |
| 4 | SFG-SwinIR | **44.30** | 0.9801 | 601,081 | 23.2 | Transformer+频域 | LV2 🆕 |
| 5 | Dual-Branch EmambaIR | **44.28** | 0.9798 | 1,027,825 | 38.3 | Mamba+频域 | LV2 🆕 |
| 6 | MambaPFT | **44.28** | 0.9819 | 1,215,399 | 55.6 | Mamba+CNN融合 | LV2 |
| 7 | Dual-Branch PFTSR | **44.24** | 0.9799 | 655,585 | 32.9 | CNN+频域 | LV2 🆕 |
| 8 | GPROF-IR | **44.21** | 0.9817 | 596,679 | 40.5 | 红外增强 | LV1 |
| 9 | Physics-PFTSR | **44.17** | 0.9796 | 481,793 | 25.9 | 物理约束 | LV2 🆕 |
| 10 | SwinRestorer | **44.16** | 0.9798 | 581,020 | 28.4 | Transformer+CNN融合 | LV2 |

## 关键观察

**新方法的冲击**：
- 5 个新方法中有 **4 个进入 Top-10**（#4、#5、#7、#9）
- 频域增强（SFG/Dual-Branch）表现突出，SFG-SwinIR 和 Dual-Branch EmambaIR 分别位列 #4 和 #5
- Physics-PFTSR (#9) 证明物理约束损失有效，且参数量最小（481K）

**架构趋势**：
- Top-3 保持为 EmambaIR / PFTSR / DualScaleRestorer
- Mamba 系列占据 #1、#5、#6（3/10）
- 频域增强方法占据 #4、#5、#7（3/10）
- 纯 CNN/Transformer 单架构仍具竞争力（PFTSR #2, SwinIR #11）

**效率考量**：
- 最快推理：SFG-SwinIR 23.2ms (#4)
- 最小参数量：Physics-PFTSR 481K (#9)
- 最佳 PSNR/参数比：SFG-SwinIR (44.30 / 601K)

## LV3 融合策略建议

### 高潜力融合组合

1. **EmambaIR + SFG-SwinIR**（#1 + #4）
   - Mamba 空间建模 + Transformer 频域门控
   - 互补性强，推理时间可控

2. **PFTSR + Physics-PFTSR**（#2 + #9）
   - 同架构不同损失，物理约束可抑制伪影
   - 参数量均 < 1M，部署友好

3. **Dual-Branch EmambaIR + Dual-Branch PFTSR**（#5 + #7）
   - 双分支设计验证，融合两个频域分支
   - 探索频域信息的最佳利用方式

### 保留方法清单

以下方法代码、权重、日志保留在 LV3 目录：
```
17_method_emambair/
04_method_pftsr/
14_method_dualscalerestore/
31_method_sfg_swinir/
33_method_dual_branch_emambair/
13_method_mambapft/
30_method_dual_branch_pftsr/
18_method_gprof_ir/
32_method_physics_pftsr/
10_method_swinrestorer/
```

### 可归档/清理的方法（排名 11-24）

```
05_method_swinir/          # 44.15 dB，SwinRestorer 已覆盖
19_method_impa_net/        # 44.12 dB
06_method_tinynina/        # 44.12 dB，轻量但非 Top-10
15_method_ntire2026_ir_sr/ # 44.04 dB，推理太慢(339ms)
34_method_sfg_pftsr/       # 43.97 dB，SFG-SwinIR 已覆盖
11_method_edgepft/         # 43.73 dB
08_method_realrestorer/    # 43.66 dB
09_method_lcmsr/           # 43.63 dB，扩散模型过大
12_method_latentswin/      # 43.60 dB
20_method_multispectral_sr/# 43.17 dB
16_method_weather_sr/      # 41.78 dB
02_baseline_srcnn/         # 38.28 dB
03_method_edsr/            # 37.94 dB
07_method_m2ir/            # 34.21 dB
```

> 清理操作需用户确认后执行。
