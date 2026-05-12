# 卫星数据超分辨率技术论文汇总

**整理日期**: 2026-05-11
**搜索来源**: arXiv API + Serper 学术搜索
**时间范围**: 过去120-180天
**主题**: 卫星数据超分辨率技术、遥感图像增强、气象卫星

---

## 一、核心论文（与 FY-4B 直接相关）

### 1. NTIRE 2026 遥感红外图像超分辨率挑战赛

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2604.21312 |
| **标题** | The First Challenge on Remote Sensing Infrared Image Super-Resolution at NTIRE 2026: Benchmark Results and Method Overview |
| **作者** | Kai Liu, Haoyang Yue, Zeli Lin, Zheng Chen, Jingkai Wang, et al. (68位作者) |
| **发布日期** | 2026-04-23 |
| **分类** | cs.CV, cs.AI |
| **引用数** | ~19 |

**论文简介**: 本文介绍了 NTIRE 2026 遥感红外图像超分辨率 (x4) 挑战赛。挑战赛旨在从低分辨率 (LR) 红外图像输入中恢复高分辨率 (HR) 红外图像，放大因子为 x4。这是遥感领域红外图像 SR 的首个大规模挑战赛。

**技术精华**:
- **任务**: 红外图像 x4 超分辨率
- **基准**: 双三次下采样生成 LR 输入
- **评估指标**: PSNR, SSIM, LPIPS
- **数据集**: 遥感红外图像，与可见光遥感不同的光谱特性
- **参赛方法**: CNN、Transformer、混合架构等多种深度学习方法

**GitHub 代码**: 无官方仓库（挑战赛基准）

**论文地址**:
- arXiv: https://arxiv.org/abs/2604.21312
- PDF: https://arxiv.org/pdf/2604.21312

---

### 2. 全球天气预报超分辨率

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2409.11502 |
| **标题** | Super Resolution on Global Weather Forecasts |
| **作者** | Lawrence Zhang, Adam Yang, Rodz Andrie Amor, Bryan Zhang, Dhruv Rao |
| **发布日期** | 2024-09-17 |
| **分类** | cs.LG, physics.ao-ph |
| **引用数** | ~3 |

**论文简介**: 研究基于神经网络将低分辨率气象图像超分辨率到高分辨率的方法。气象预报对于日常活动规划到灾害响应都至关重要。由于气象系统的混沌性，传统方法难以长期准确预测。本文提出端到端深度学习方法学习 LR 到 HR 的映射。

**技术精华**:
- **核心任务**: 气象图像超分辨率
- **应用场景**: 全球天气预报
- **方法**: 端到端深度学习映射
- **特点**: 与 FY-4B 卫星气象应用高度相关
- **创新点**: 将 SR 技术应用于气象领域

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2409.11502
- PDF: https://arxiv.org/pdf/2409.11502

---

### 3. 多光谱卫星图像超分辨率 (CNN方法)

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2002.00580 |
| **标题** | Super-resolution of multispectral satellite images using convolutional neural networks |
| **作者** | M. U. Müller, N. Ekhtiari, R. M. Almeida, C. Rieke |
| **发布日期** | 2020-02-03 |
| **分类** | eess.IV, cs.CV |
| **引用数** | 113 |

**论文简介**: 使用卷积神经网络处理多光谱卫星图像的超分辨率问题。研究指出，虽然大多数研究集中在 RGB 彩色照片的处理上，但针对多波段分析卫星图像的研究较少。本文探索了自编码器和残差网络等 CNN 架构在卫星图像 SR 上的应用。

**技术精华**:
- **网络架构**: 自编码器、残差网络
- **数据特点**: 多波段卫星图像（非 RGB）
- **方法论**: 与 FY-4B 多通道特性高度相关
- **贡献**: 为遥感图像 SR 提供基准

**GitHub 代码**: 需进一步确认

**论文地址**:
- arXiv: https://arxiv.org/abs/2002.00580
- PDF: https://arxiv.org/pdf/2002.00580

---

## 二、遥感图像处理相关论文

### 4. 卫星-无人机-地面跨视图3D重建

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07978 |
| **标题** | Seeing Across Skies and Streets: Feedforward 3D Reconstruction from Satellite, Drone, and Ground Images |
| **作者** | Qiwei Wang, Zhongyao Tuo, Xianghui Ze, Yujiao Shi |
| **发布日期** | 2026-05-08 |
| **分类** | cs.CV |

**论文简介**: 跨视图定位经典问题是确定地面图像在卫星瓦片上的位置。本文突破传统 3-DoF 估计限制（x, y 位置和偏航角），提出从卫星、无人机和地面图像进行前馈式 3D 重建的方法。

**技术精华**:
- **创新点**: 突破传统 3-DoF 限制，实现完整 3D 重建
- **输入**: 卫星、无人机、地面多视角图像
- **输出**: 前馈式 3D 重建
- **应用**: 跨视图定位、3D 重建
- **方法**: 利用 UAV 作为中间视图桥接卫星和地面视角

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07978
- PDF: https://arxiv.org/pdf/2605.07978

---

### 5. 卫星DSM重建 (RPC感知深度微调)

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07264 |
| **标题** | Sat3R: Satellite DSM Reconstruction via RPC-Aware Depth Fine-tuning |
| **作者** | Qiaoyi Yang, Chaoyi Zhou, Xi Liu, et al. |
| **发布日期** | 2026-05-08 |
| **分类** | cs.CV |

**论文简介**: 从卫星图像进行精确的数字表面模型 (DSM) 重建对于灾害响应、城市规划和大规模地理测绘等应用至关重要。Sat3R 通过 RPC（有理多项式系数）感知深度微调方法实现高精度的 DSM 重建。

**技术精华**:
- **核心任务**: 卫星图像 DSM 重建
- **关键技术**: RPC 感知深度微调
- **应用场景**: 灾害响应、城市规划、地理测绘
- **创新**: 利用 RPC 模型的几何约束

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07264
- PDF: https://arxiv.org/pdf/2605.07264

---

### 6. 卫星表面重建 (高斯溅射)

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07181 |
| **标题** | SatSurfGS: Generalizable 2D Gaussian Splatting for Sparse-View Satellite Surface Reconstruction |
| **作者** | Min Chen, Wei Guo, Bin Wang, et al. |
| **发布日期** | 2026-05-08 |
| **分类** | cs.CV |

**论文简介**: 稀疏视角卫星图像表面重建面临重大挑战。SatSurfGS 提出基于 2D 高斯溅射的可泛化方法，解决卫星成像条件下多视图匹配可靠性空间异质性问题。

**技术精华**:
- **方法**: 2D 高斯溅射 (Gaussian Splatting)
- **特点**: 可泛化、稀疏视角处理
- **挑战**: 卫星成像条件下的多视图匹配
- **创新**: 解决空间异质性问题

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07181
- PDF: https://arxiv.org/pdf/2605.07181

---

### 7. 卫星降水反演红外增强

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07167 |
| **标题** | GPROF-IR: An Improved Single-Channel Infrared Precipitation Retrieval for Merged Satellite Precipitation Products |
| **作者** | Simon Pfreundschuh, Christian D. Kummerow, Jackson Tan, George J. Huffman |
| **发布日期** | 2026-05-08 |
| **分类** | physics.ao-ph |

**论文简介**: 当前降水产品（如 IMERG, GSMAP, CMORPH）结合了被动微波 (PMW) 和红外 (IR) 观测的卫星估计。GPROF-IR 提出改进的单通道红外降水反演方法，解决不同传感器信息内容差异导致的估计不一致问题。

**技术精华**:
- **任务**: 卫星降水反演
- **方法**: 单通道红外增强
- **应用**: 气象预报、降水估计
- **与 FY-4B 关系**: FY-4B 的 IR 通道可借鉴此方法

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07167
- PDF: https://arxiv.org/pdf/2605.07167

---

### 8. 遥感VLM尺度条件化

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07562 |
| **标题** | Beyond GSD-as-Token: Continuous Scale Conditioning for Remote Sensing VLMs |
| **作者** | Song Zhang, Yanlong Chen, Yilin Li, et al. |
| **发布日期** | 2026-05-08 |
| **分类** | cs.CV |

**论文简介**: 遥感视觉语言模型 (RS-VLMs) 与自然图像对应物存在根本性不匹配：同一地理目标在地表采样时表现出截然不同的视觉证据（GSD差异）。本文提出连续尺度条件化方法解决这一问题。

**技术精华**:
- **问题**: 遥感图像尺度变化大（GSD差异）
- **方法**: 连续尺度条件化
- **创新**: 超越 GSD-as-Token 范式
- **应用**: 提升 RS-VLM 泛化能力

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07562
- PDF: https://arxiv.org/pdf/2605.07562

---

### 9. 跨视图地理定位

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07099 |
| **标题** | InfoGeo: Information-Theoretic Object-Centric Learning for Cross-View Generalizable UAV Geo-Localization |
| **作者** | Hongyang Zhang, Maonnan Wang, Ziyao Wang, et al. |
| **发布日期** | 2026-05-08 |
| **分类** | cs.CV |

**论文简介**: 跨视图地理定位 (CVGL) 对于 GPS 拒止环境中的精确定位和导航至关重要。InfoGeo 提出基于信息论的目标中心学习方法，提升跨视图泛化能力。

**技术精华**:
- **任务**: 无人机-卫星跨视图定位
- **方法**: 信息论目标中心学习
- **挑战**: 区域纹理和天气条件变化导致的域偏移
- **创新**: 对象级别的特征对齐

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07099
- PDF: https://arxiv.org/pdf/2605.07099

---

### 10. PACE卫星AOD估计基础模型

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.00678 |
| **标题** | Foundation AI Models for Aerosol Optical Depth Estimation from PACE Satellite Data |
| **作者** | Zahid Hassan Tushar, Sanjay Purushotham |
| **发布日期** | 2026-05-01 |
| **分类** | cs.CV |

**论文简介**: 气溶胶光学厚度 (AOD) 检索对地球观测至关重要，支持空气质量和气候变化研究等应用。传统物理方法难以处理复杂大气条件。本文提出基于 PACE 卫星数据的基础 AI 模型进行 AOD 估计。

**技术精华**:
- **任务**: AOD 估计（卫星遥感）
- **数据**: PACE 卫星数据
- **方法**: 基础 AI 模型
- **应用**: 空气质量管理、气候研究

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.00678
- PDF: https://arxiv.org/pdf/2605.00678

---

### 11. 遥感岩石分类基准

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07640 |
| **标题** | LithoBench: Benchmarking Large Multimodal Models for Remote-Sensing Lithology Interpretation |
| **作者** | Jun Wang, Fengpeng Li, Hang Dong, et al. |
| **发布日期** | 2026-05-08 |
| **分类** | cs.CV, cs.AI |

**论文简介**: 遥感岩石地层解释是地质调查、矿产勘探和区域地质填图的基础。LithoBench 为遥感岩石解释建立评估基准，测试大型多模态模型的细粒度理解能力。

**技术精华**:
- **任务**: 遥感岩石地层解释
- **方法**: 大型多模态模型评估
- **应用**: 地质调查、矿产勘探
- **意义**: 建立遥感细粒度分类基准

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07640
- PDF: https://arxiv.org/pdf/2605.07640

---

## 三、气象/雷达相关论文

### 12. 气象感知雷达临近预报

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2604.24224 |
| **标题** | IMPA-Net: Meteorology-Aware Multi-Scale Attention and Dynamic Loss for Extreme Convective Radar Nowcasting |
| **作者** | Haofei Cui, Guangxin He, Juanzhen Sun, et al. |
| **发布日期** | 2026-04-27 |
| **分类** | cs.LG |

**论文简介**: 对流降水的短时预报对强天气预报至关重要。深度学习模型使用像素级误差指标训练时容易过度平滑预测结果。IMPA-Net 提出气象感知多尺度注意力机制和动态损失函数解决这一问题。

**技术精华**:
- **任务**: 对流降水临近预报（雷达）
- **创新**: 气象感知多尺度注意力
- **损失函数**: 动态损失设计
- **应用**: 强天气预报
- **借鉴价值**: 多尺度特征提取可用于 FY-4B 图像处理

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2604.24224
- PDF: https://arxiv.org/pdf/2604.24224

---

### 13. ERA5湍流预测误差分析

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.07981 |
| **标题** | Learning from Translation: Seasonal Errors and Feature Importance of the ERA5 Turbulence Predictions |
| **作者** | Arial Tolentino, Markus Petters, Luat T. Vuong |
| **发布日期** | 2026-05-08 |
| **分类** | physics.ao-ph, physics.optics |

**论文简介**: 研究 ERA5 湍流预测的季节性误差和特征重要性。湍流在局部由测量表征，但由非局部的能量级联引起。本文分析遥感观测在湍流建模中的作用。

**技术精华**:
- **任务**: 气象湍流预测
- **数据**: ERA5 再分析数据
- **方法**: 误差分析和特征重要性
- **应用**: 天气预报模型改进

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.07981
- PDF: https://arxiv.org/pdf/2605.07981

---

## 四、图像重建/增强方法论文

### 14. EmambaIR: 事件引导图像重建

| 属性 | 内容 |
|------|------|
| **arXiv ID** | 2605.08073 |
| **标题** | EmambaIR: Efficient Visual State Space Model for Event-guided Image Reconstruction |
| **作者** | Wei Yu, Yunhang Qian |
| **发布日期** | 2026-05-08 |
| **分类** | cs.CV, cs.AI |

**论文简介**: 事件相机图像重建方法主要依赖 CNN 和 ViT，但存在局限性：CNN 难以捕获全局特征相关性，ViT 计算开销大。EmambaIR 提出基于 Mamba 状态空间模型的高效事件引导图像重建方法。

**技术精华**:
- **方法**: 视觉状态空间模型 (Mamba)
- **优势**: 兼顾全局建模和计算效率
- **任务**: 事件引导图像重建
- **借鉴价值**: 可探索 Mamba 在卫星图像 SR 中的应用

**GitHub 代码**: 无官方仓库

**论文地址**:
- arXiv: https://arxiv.org/abs/2605.08073
- PDF: https://arxiv.org/pdf/2605.08073

---

## 五、论文汇总表

| 序号 | arXiv ID | 标题 | 日期 | 分类 | GitHub | 与FY-4B相关性 |
|------|----------|------|------|------|--------|---------------|
| 1 | 2604.21312 | NTIRE 2026 遥感红外 SR 挑战赛 | 2026-04 | cs.CV | 无 | **极高** |
| 2 | 2409.11502 | 全球天气预报超分辨率 | 2024-09 | cs.LG | 无 | **极高** |
| 3 | 2002.00580 | 多光谱卫星图像 SR (CNN) | 2020-02 | eess.IV | 待确认 | **高** |
| 4 | 2605.07978 | 跨视图3D重建 | 2026-05 | cs.CV | 无 | 中 |
| 5 | 2605.07264 | Sat3R 卫星 DSM 重建 | 2026-05 | cs.CV | 无 | 中 |
| 6 | 2605.07181 | SatSurfGS 表面重建 | 2026-05 | cs.CV | 无 | 中 |
| 7 | 2605.07167 | GPROF-IR 红外降水增强 | 2026-05 | physics | 无 | **高** |
| 8 | 2605.07562 | 遥感 VLM 尺度条件化 | 2026-05 | cs.CV | 无 | 低 |
| 9 | 2605.07099 | InfoGeo 跨视图定位 | 2026-05 | cs.CV | 无 | 中 |
| 10 | 2605.00678 | PACE 卫星 AOD 估计 | 2026-05 | cs.CV | 无 | 中 |
| 11 | 2605.07640 | LithoBench 岩石分类 | 2026-05 | cs.CV | 无 | 低 |
| 12 | 2604.24224 | IMPA-Net 雷达临近预报 | 2026-04 | cs.LG | 无 | **高** |
| 13 | 2605.07981 | ERA5 湍流预测误差 | 2026-05 | physics | 无 | 中 |
| 14 | 2605.08073 | EmambaIR 事件重建 | 2026-05 | cs.CV | 无 | 中 |

---

## 六、重点推荐（技术借鉴价值排序）

### 第一梯队（直接相关）

1. **2604.21312 - NTIRE 2026 遥感红外 SR 挑战赛**
   - 专门针对遥感红外图像超分辨率
   - 与 FY-4B 红外通道 (CH07/CH08) 直接相关
   - 包含最新方法比较和基准

2. **2409.11502 - 全球天气预报超分辨率**
   - 气象图像超分辨率端到端方法
   - 与 FY-4B 气象应用高度一致
   - 可借鉴其网络架构设计

3. **2002.00580 - 多光谱卫星图像 SR**
   - CNN 方法处理多波段卫星图像
   - 为 FY-4B 多通道处理提供参考
   - 引用数高(113)，方法成熟

### 第二梯队（方法借鉴）

4. **2604.24224 - IMPA-Net**
   - 气象感知多尺度注意力机制
   - 动态损失函数设计
   - 可借鉴用于 FY-4B 图像增强

5. **2605.07167 - GPROF-IR**
   - 红外图像增强方法
   - 与 FY-4B IR 通道直接相关

6. **2605.08073 - EmambaIR**
   - Mamba 状态空间模型
   - 高效全局建模
   - 可探索用于卫星图像 SR

---

## 七、GitHub 代码情况汇总

| 状态 | 数量 | 说明 |
|------|------|------|
| **有官方仓库** | 0 | 本次搜索结果均无公开代码 |
| **需联系作者** | 1 | 2002.00580 可能有代码 |
| **无仓库** | 13 | 大多数论文未公开代码 |

**建议**:
1. 访问论文页面查看是否有最新代码更新
2. 联系论文作者请求代码
3. 参考已开源的相关工作（如 SwinIR, ESRGAN 等经典方法）

---

## 八、搜索说明

- **搜索工具**: github-arxiv-v1 skill（arXiv API + Serper 学术搜索）
- **搜索关键词**: satellite super resolution, remote sensing, meteorological, infrared, weather forecast
- **时间范围**: 2025-2026年（120-365天）
- **搜索限制**: arXiv API 有请求频率限制，本报告已整合多次搜索结果

---

*本文件由 github-arxiv-v1 skill 自动整理*
*整理时间: 2026-05-11*
