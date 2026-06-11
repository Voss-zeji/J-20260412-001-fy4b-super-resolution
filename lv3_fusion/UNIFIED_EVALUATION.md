# FY-4B 超分辨率 -- 统一评估汇总 (200 epoch)

**生成时间**: 2026-06-07 17:45:09  
**评估口径**: 统一取 best checkpoint PSNR（如无则取 final epoch），排除 Bicubic(inf)  
**有效方法**: 24 | **跳过**: 1  

## 完整排名 (PSNR 降序)

| 排名 | 方法 | PSNR (dB) | SSIM | Epoch | 参数量 | 推理(ms) | 运行(分) | 数据来源 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 17_method_emambair | 44.42 | 0.9803 | 178 | 890,889 | 60.1 | 486 | best_checkpoint |
| 2 | 04_method_pftsr | 44.35 | 0.9816 | 200 | 1,000,103 | 39.6 | 84 | final_epoch |
| 3 | 14_method_dualscalerestore | 44.34 | 0.9803 | 200 | 1,586,415 | 32.7 | 81 | final_epoch |
| 4 | 31_method_sfg_swinir | 44.30 | 0.9801 | 176 | 601,081 | 23.2 | ? | best_checkpoint |
| 5 | 33_method_dual_branch_emambair | 44.28 | 0.9798 | 179 | 1,027,825 | 38.3 | ? | best_checkpoint |
| 6 | 13_method_mambapft | 44.28 | 0.9819 | 200 | 1,215,399 | 55.6 | 94 | final_epoch |
| 7 | 30_method_dual_branch_pftsr | 44.24 | 0.9799 | 184 | 655,585 | 32.9 | ? | best_checkpoint |
| 8 | 18_method_gprof_ir | 44.21 | 0.9817 | 190 | 596,679 | 40.5 | 483 | best_checkpoint |
| 9 | 32_method_physics_pftsr | 44.17 | 0.9796 | 180 | 481,793 | 25.9 | ? | best_checkpoint |
| 10 | 10_method_swinrestorer | 44.16 | 0.9798 | 200 | 581,020 | 28.4 | 80 | final_epoch |
| 11 | 05_method_swinir | 44.15 | 0.9799 | 200 | 406,441 | 20.7 | 83 | final_epoch |
| 12 | 19_method_impa_net | 44.12 | 0.9805 | 163 | 1,116,663 | 9.0 | 475 | best_checkpoint |
| 13 | 06_method_tinynina | 44.12 | 0.9815 | 200 | 45,418 | 18.8 | 83 | final_epoch |
| 14 | 15_method_ntire2026_ir_sr | 44.04 | 0.9812 | 143 | 1,168,924 | 339.1 | 495 | best_checkpoint |
| 15 | 34_method_sfg_pftsr | 43.97 | 0.9788 | 200 | 703,233 | 21.7 | ? | best_checkpoint |
| 16 | 11_method_edgepft | 43.73 | 0.9761 | 200 | 94,287 | 30.1 | 79 | final_epoch |
| 17 | 08_method_realrestorer | 43.66 | 0.9768 | 200 | 1,043,797 | 27.0 | 82 | final_epoch |
| 18 | 09_method_lcmsr | 43.63 | 0.9766 | 200 | 2,753,833 | 23.8 | 82 | final_epoch |
| 19 | 12_method_latentswin | 43.60 | 0.9766 | 200 | 5,058,785 | 64.4 | 83 | final_epoch |
| 20 | 20_method_multispectral_sr | 43.17 | 0.9782 | 197 | 778,753 | 4.4 | 464 | best_checkpoint |
| 21 | 16_method_weather_sr | 41.78 | 0.9706 | 185 | ? | ? | 486 | best_checkpoint |
| 22 | 02_baseline_srcnn | 38.28 | 0.9467 | 200 | 8,129 | 18.3 | 82 | final_epoch |
| 23 | 03_method_edsr | 37.94 | 0.9578 | 200 | 1,367,553 | 21.7 | 82 | final_epoch |
| 24 | 07_method_m2ir | 34.21 | 0.8742 | 200 | 328,993 | 172.4 | 95 | final_epoch |

## Top-10 方法

### 1. 17_method_emambair
- **PSNR**: 44.42 dB
- **SSIM**: 0.9803
- **参数量**: 890,889
- **推理时间**: 60.1 ms
- **训练时长**: 486 分钟
- **数据来源**: best_checkpoint

### 2. 04_method_pftsr
- **PSNR**: 44.35 dB
- **SSIM**: 0.9816
- **参数量**: 1,000,103
- **推理时间**: 39.6 ms
- **训练时长**: 84 分钟
- **数据来源**: final_epoch

### 3. 14_method_dualscalerestore
- **PSNR**: 44.34 dB
- **SSIM**: 0.9803
- **参数量**: 1,586,415
- **推理时间**: 32.7 ms
- **训练时长**: 81 分钟
- **数据来源**: final_epoch

### 4. 31_method_sfg_swinir
- **PSNR**: 44.30 dB
- **SSIM**: 0.9801
- **参数量**: 601,081
- **推理时间**: 23.2 ms
- **训练时长**: 0 分钟
- **数据来源**: best_checkpoint

### 5. 33_method_dual_branch_emambair
- **PSNR**: 44.28 dB
- **SSIM**: 0.9798
- **参数量**: 1,027,825
- **推理时间**: 38.3 ms
- **训练时长**: 0 分钟
- **数据来源**: best_checkpoint

### 6. 13_method_mambapft
- **PSNR**: 44.28 dB
- **SSIM**: 0.9819
- **参数量**: 1,215,399
- **推理时间**: 55.6 ms
- **训练时长**: 94 分钟
- **数据来源**: final_epoch

### 7. 30_method_dual_branch_pftsr
- **PSNR**: 44.24 dB
- **SSIM**: 0.9799
- **参数量**: 655,585
- **推理时间**: 32.9 ms
- **训练时长**: 0 分钟
- **数据来源**: best_checkpoint

### 8. 18_method_gprof_ir
- **PSNR**: 44.21 dB
- **SSIM**: 0.9817
- **参数量**: 596,679
- **推理时间**: 40.5 ms
- **训练时长**: 483 分钟
- **数据来源**: best_checkpoint

### 9. 32_method_physics_pftsr
- **PSNR**: 44.17 dB
- **SSIM**: 0.9796
- **参数量**: 481,793
- **推理时间**: 25.9 ms
- **训练时长**: 0 分钟
- **数据来源**: best_checkpoint

### 10. 10_method_swinrestorer
- **PSNR**: 44.16 dB
- **SSIM**: 0.9798
- **参数量**: 581,020
- **推理时间**: 28.4 ms
- **训练时长**: 80 分钟
- **数据来源**: final_epoch

## 全部方法柱状图数据



## 跳过/无效的方法

01_baseline_bicubic