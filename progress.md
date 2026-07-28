# 进度记录 - FY-4B 超分辨率研究

## 任务信息
- **任务ID**: J-20260412-001
- **创建日期**: 2026-04-12
- **状态**: 进行中
- **方法论**: 三层实验架构 + 退化模拟（v2 修正）

## 执行日志

### 2026-04-12 — 初始搭建
- [x] 创建任务目录结构，从 GPU 服务器复制代码
- [x] 设计三层实验架构（lv1_macro / lv2_micro / lv3_fusion）
- [x] 初始提交 (commit 323bccb)

### 2026-04-14 ~ 2026-04-26 — 方法扩展
- [x] 5 → 9 → 29+ 种超分方法收集与适配
- [x] CH07/CH08 定标批量处理

### 2026-05-23 ~ 2026-05-25 — LV3 深度训练（CH07, 200 epoch）
- [x] 24 种方法统一 200 epoch 训练
- [x] Top-10 PSNR 全部 > 44 dB（基于 CH07 真实配对）
- [x] 3 个最佳 checkpoint 保存（EmambaIR, SFGSwinIR, PhysicsPFTSR）

### 2026-06-07 — Top-10 选定
- [x] TOP10_SELECTION.md 生成
- [x] 3 组融合策略规划

### 2026-06-08 — 产品生成 & 数据审查
- [x] CH08 数据准备（19184 patches）
- [x] CH07 全盘 SR 产品生成（3 模型 × 10 样本）
- [x] CH08 zero-shot 评估（43.6 dB）— ⚠️ 后续确认为无效（HR 是内插）
- [x] 发现关键问题：CH08 2000M 为 4000M 内插产物，非独立观测
- [x] program.md 修正：更新数据事实、设计退化方案

### 2026-06-10 — 退化校准 + CH08 验证（远程已执行，本地同步）
- [x] 退化模型校准完成：PSF σ=0.384（高斯核），噪声 σ=0.4355 K
- [x] 优化 RMSE=0.5275，比双三次下采样提升 48.7%
- [x] CH07 合成测试集评估：6 模型排名与真实配对 100% 一致
- [x] CH08 自监督微调（EmambaIR，50 epoch）：Val PSNR=41.39 dB
- [x] CH08 真实 4000M SR 推理：10 个产品生成，代理 PSNR=34.68 dB
- [x] CH08 产品保存至 `/root/autodl-tmp/products/ch08/`

### 2026-06-23 — 对比可视化生成
- [x] 生成 5 张综合对比图（CH07/CH08 对比、退化验证、PSNR 排名、代理 PSNR）
- [x] 结果同步到本地 `results/`

### 2026-06-23 — LV3 融合评估
- [x] EmambaIR + SFGSwinIR 集成评估：PSNR=43.62 dB
- [x] 结论：简单平均集成未超过 EmambaIR 单独（43.65 dB），建议直接使用 EmambaIR 作为最终模型
- [x] 融合结果保存至 `lv3_fusion/fusion_result.json`
- [ ] 清理低分方法（排名 11-24，可选）

### 2026-07-17 — 远程代码回同步与状态核验
- [x] 已登录 GPU 服务器并核验远端项目：`/root/jobs/J-20260412-001-fy4b-super-resolution`
- [x] 本地与远端均基于提交 `96e8f83`；远端无正在运行的训练、微调或推理任务
- [x] 校验并保留本地已有的两份未提交脚本（哈希与远端一致）：`scripts/generate_comparison_figures.py`、`scripts/lv3_fusion.py`
- [x] 同步远端缺失结果：`lv3_fusion/fusion_result.json`（SHA-256 已校验）
- [ ] 清理低分方法（排名 11-24，可选）
