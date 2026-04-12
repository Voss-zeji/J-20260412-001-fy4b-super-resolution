# FY-4B AGRI L1 数据定标处理工具

## 概述

本工具用于将FY-4B卫星AGRI传感器的L1级数据中的DN值（Digital Number）转换为实际物理量（反射率或亮温）。

## FY-4B数据定标原理

### 数据文件结构

FY-4B AGRI L1 HDF文件包含以下关键数据集：

| 数据集路径 | 说明 | 形状 | 数据类型 |
|-----------|------|------|---------|
| `Data/NOMChannel01-08` | DN值 | (5496, 5496) | uint16 |
| `Calibration/CALChannel01-08` | 查找表 | (4096,) | float32 |
| `Calibration/CALIBRATION_COEF(SCALE+OFFSET)` | 线性定标系数 | (8, 2) | float32 |
| `Calibration/ESUN` | 太阳辐照度 | (8,) | float32 |

### 波段信息

| 波段 | 名称 | 类型 | 波长 | 物理量范围 |
|------|------|------|------|-----------|
| Channel01 | VIS0.47 | 反射率 | 0.47μm | 0.0 - 1.1431 |
| Channel02 | VIS0.65 | 反射率 | 0.65μm | 0.0 - 1.1431 |
| Channel03 | VIS0.83 | 反射率 | 0.83μm | 0.0 - 1.1431 |
| Channel04 | NIR1.38 | 反射率 | 1.38μm | 0.0 - 1.1431 |
| Channel05 | NIR1.61 | 反射率 | 1.61μm | 0.0 - 1.1431 |
| Channel06 | NIR2.25 | 反射率 | 2.25μm | 0.0 - 1.1431 |
| Channel07 | IR3.90 | 亮温 | 3.90μm | 199.99 - 404.99 K |
| Channel08 | IR7.00 | 亮温 | 7.00μm | 190.00 - 341.00 K |

**注意**: Channel07和Channel08都是中红外通道，用于测量地球表面的热辐射。

### 定标方法

#### 1. 查找表定标（推荐）

FY-4B采用查找表（LUT）方式进行定标，这是精度最高的方法：

```
物理量 = CALTable[DN]
```

其中：
- DN值范围: 0 - 4095（12位有效数据）
- 查找表索引: 直接使用DN值作为索引
- 填充值: 65535（转换为np.nan）

#### 2. 线性定标

也可使用线性公式进行快速定标：

```
物理量 = DN × Scale + Offset
```

定标系数示例：
- Channel01-06（反射率）: Scale ≈ 0.000284, Offset ≈ -0.0215
- Channel07-08（亮温）: Scale为负值，Offset为正值

**注意**: 线性定标对于红外波段（Channel07-08）精度较低，建议使用查找表方法。

## 使用方法

### 1. 作为Python模块使用

```python
from fy4b_calibration import FY4BCalibrator

# 创建定标器
calibrator = FY4BCalibrator('/path/to/FY4B_file.HDF')

# 打印定标信息
calibrator.print_calibration_info()

# 使用查找表定标单个波段
reflectance = calibrator.calibrate_with_lut('Channel01')

# 使用线性定标
brightness_temp = calibrator.calibrate_linear('Channel07')

# 定标所有波段
all_bands = calibrator.calibrate_all_bands(method='lut')
```

### 2. 命令行工具

```bash
# 处理单个文件
python fy4b_calibration.py -i /path/to/FY4B_file.HDF -o ./output

# 批量处理
python fy4b_calibration.py -i /path/to/input_dir -o ./output --batch

# 使用线性定标方法
python fy4b_calibration.py -i /path/to/FY4B_file.HDF -o ./output --method linear
```

### 3. 运行演示脚本

```bash
python demo_calibration.py
```

演示脚本将：
1. 分析文件结构和定标信息
2. 对比两种定标方法的效果
3. 生成可视化图像
4. 输出定标后的数据

## 输出文件

定标后的数据可以保存为：
- `.npy` 格式: NumPy数组文件
- `.hdf` 格式: 包含定标数据和元数据的HDF5文件

## 注意事项

1. **填充值处理**: 原始数据中的65535为填充值，定标后会转换为np.nan
2. **内存使用**: 完整图像大小为5496×5496，每个波段约115MB（float32）
3. **定标方法选择**: 
   - 科学分析建议使用查找表（LUT）方法
   - 快速预览可使用线性方法
4. **有效数据范围**: DN值通常小于4096，超出此范围会被裁剪

## 示例输出

```
============================================================
文件: FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250322000000_...HDF
============================================================

【定标系数 (Scale, Offset)】
  Channel01 (VIS0.47): Scale=0.0002844045, Offset=-0.021536 [reflectance]
  Channel07 (IR3.90): Scale=-0.0029591168, Offset=12.118389 [brightness_temp]

【查找表范围】
  Channel01: min=0.0000, max=1.1431 [反射率]
  Channel07: min=199.9900, max=404.9900 [亮温(K)]
```

## 依赖库

- numpy
- h5py
- matplotlib (用于可视化)

## 参考资料

- FY-4B卫星AGRI传感器L1数据产品规格说明
- 国家卫星气象中心(NSMC)数据文档
