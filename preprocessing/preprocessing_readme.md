# FY-4B AGRI L1 数据定标预处理工具

## 概述

本工具用于将FY-4B卫星AGRI传感器的L1级数据中的DN值（Digital Number）转换为实际物理量（反射率或亮温）。这是卫星遥感数据预处理的关键步骤，将原始的数字化观测值转换为具有物理意义的可量化数据。

## 处理什么文件

### 输入文件

**文件类型**：FY-4B AGRI L1 级 HDF 格式数据文件

**文件名格式示例**：
```
FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250322000000_20250322001459_2000M_V0001.HDF
```

**输入路径**（代码中配置的默认路径）：
- `/root/autodl-tmp/2000M` - 2000米分辨率全圆盘数据
- `/root/autodl-tmp/4000M` - 4000米分辨率全圆盘数据

### 输入文件内部结构

| 数据集路径 | 说明 | 数据类型 | 用途 |
|-----------|------|---------|------|
| `Data/NOMChannel01-08` | DN值（原始观测数据） | uint16, (5496, 5496) | 待转换的原始数值 |
| `Calibration/CALChannel01-08` | 查找表（LUT） | float32, (4096,) | 查找表定标 |
| `Calibration/CALIBRATION_COEF(SCALE+OFFSET)` | 线性定标系数 | float32, (8, 2) | 线性定标 |
| `Calibration/ESUN` | 太阳辐照度 | float32, (8,) | 反射率计算辅助 |
| `QA/CalQualityFlag` | 定标质量标志 | - | 质量检查 |
| `QA/L1QualityFlag` | L1数据质量标志 | - | 质量检查 |

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

> **注意**：Channel01-06为可见光/近红外波段，Channel07-08为中红外波段，用于测量地球表面的热辐射。

---

## 怎么处理

### 定标原理

#### 方法1：查找表定标（推荐，精度最高）

FY-4B采用查找表（Lookup Table, LUT）方式进行定标：

```
物理量 = CALTable[DN]
```

处理步骤：
1. DN值范围：0 - 4095（12位有效数据）
2. 使用DN值直接作为索引查询查找表
3. 填充值65535转换为 `np.nan`
4. 超出范围的DN值会被裁剪到有效区间

#### 方法2：线性定标（快速预览）

使用线性公式进行快速定标：

```
物理量 = DN × Scale + Offset
```

定标系数特点：
- Channel01-06（反射率）：Scale ≈ 0.000284, Offset ≈ -0.0215
- Channel07-08（亮温）：Scale为负值，Offset为正值

> **注意**：线性定标对于红外波段（Channel07-08）精度较低，科学分析建议使用查找表方法。

### 处理流程

```
输入HDF文件
    ↓
读取定标系数和查找表
    ↓
读取DN值数据 (Data/NOMChannelXX)
    ↓
处理填充值 (65535 → np.nan)
    ↓
DN值裁剪 (0-4095范围)
    ↓
定标转换 (LUT方法或线性方法)
    ↓
生成统计信息 (min/max/mean/std)
    ↓
保存为新的HDF5文件
```

---

## 各脚本说明

### 1. fy4b_calibration.py

**核心定标模块**，提供 `FY4BCalibrator` 类，封装了所有定标功能。

**主要功能**：
- 加载HDF文件中的定标数据
- 支持两种定标方法（LUT和线性）
- 提供单波段或全波段定标
- 保存定标结果到HDF5文件

**作为Python模块使用**：

```python
from fy4b_calibration import FY4BCalibrator, save_calibrated_data

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

# 保存定标数据
save_calibrated_data(calibrator, './output', method='lut')
```

**命令行使用**：

```bash
# 处理单个文件
python fy4b_calibration.py -i /path/to/FY4B_file.HDF -o ./output

# 批量处理
python fy4b_calibration.py -i /path/to/input_dir -o ./output --batch

# 使用线性定标方法
python fy4b_calibration.py -i /path/to/FY4B_file.HDF -o ./output --method linear
```

---

### 2. batch_process.py

**通用批量处理脚本**，支持多进程并行处理所有8个波段。

**特点**：
- 支持单文件或批量目录处理
- 多进程并行加速
- 自动生成处理报告
- 可指定波段处理（默认全部）

**命令行参数**：

```bash
python batch_process.py -i <输入> -o <输出> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入文件或目录路径（必需） | - |
| `-o, --output` | 输出目录路径 | `./calibrated` |
| `-b, --batch` | 启用批量处理模式 | False |
| `-m, --method` | 定标方法：`lut` 或 `linear` | `lut` |
| `-p, --processes` | 并行进程数 | 1 |
| `--pattern` | 文件匹配模式 | `*.HDF` |
| `--compress` | 启用HDF5压缩 | True |
| `--no-compress` | 禁用HDF5压缩 | - |
| `--bands` | 指定处理波段，如 `01,02,07` | `all` |

**使用示例**：

```bash
# 处理单个文件
python batch_process.py -i /data/FY4B_2000M/20250322.HDF -o ./output

# 批量处理目录
python batch_process.py -i /data/FY4B_2000M/ -o ./output --batch

# 多进程批量处理 (使用4个进程)
python batch_process.py -i /data/FY4B_2000M/ -o ./output --batch -p 4

# 只处理特定波段时间的数据
python batch_process.py -i /data/FY4B_2000M/ -o ./output --batch --pattern "*20250322*.HDF"
```

---

### 3. batch_calibrate_ch78.py

**专用批量处理脚本**，仅处理 Channel07 和 Channel08（红外波段）。

**特点**：
- 针对红外波段优化
- 分别保存到 CH07 和 CH08 两个目录
- 使用多进程并行处理
- 自动处理 2000M 和 4000M 两种分辨率数据

**默认配置**：

```python
input_dirs = {
    '2000M': '/root/autodl-tmp/2000M',
    '4000M': '/root/autodl-tmp/4000M',
}
output_base = '/root/autodl-tmp/Calibration-FY4B'
```

**输出目录结构**：
```
/root/autodl-tmp/Calibration-FY4B/
├── 2000M/
│   ├── CH07/
│   └── CH08/
└── 4000M/
    ├── CH07/
    └── CH08/
```

**使用方式**：

```bash
python batch_calibrate_ch78.py
```

---

### 4. demo_calibration.py

**演示脚本**，展示定标过程并生成可视化图像。

**演示内容**：
1. 分析文件结构和定标信息
2. 对比两种定标方法的效果差异
3. 生成查找表特性曲线图
4. 保存定标后的数据样本

**输出文件**（保存到 `./preprocessing/output/`）：
- `calibration_comparison.png` - 两种定标方法对比图
- `lookup_tables.png` - 查找表曲线图
- `channel01_full_disk.png` - 全圆盘反射率图像
- `channel01_calibrated.npy` - 定标后的NumPy数组

**使用方式**：

```bash
python demo_calibration.py
```

---

## 输出文件说明

### 输出文件名

将原始文件名中的 `_NOM_` 替换为 `_CAL_`：

```
输入：FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250322000000_...HDF
输出：FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_CAL_20250322000000_...HDF
```

### 输出文件内部结构

| 数据集/组 | 说明 |
|-----------|------|
| `CalibratedData/Channel01-08` | 定标后的物理量数据 |
| `Metadata` | 原始文件的元数据属性 |
| `CalibrationInfo/CALIBRATION_COEF` | 定标系数备份 |
| `CalibrationInfo/ESUN` | 太阳辐照度 |

### 数据集属性

每个波段数据集包含以下属性：
- `band_name` - 波段名称（如 `IR3.90`）
- `wavelength` - 波长（如 `3.90μm`）
- `type` - 数据类型（`reflectance` 或 `brightness_temperature`）
- `calibration_method` - 使用的定标方法
- `fill_value` - 填充值（`np.nan`）
- `min/max/mean/std` - 统计信息

### 输出路径

- **全波段处理**：由命令行参数 `-o` 指定，默认 `./calibrated`
- **CH07/CH08 专用处理**：`/root/autodl-tmp/Calibration-FY4B/`
- **演示输出**：`./preprocessing/output/`

---

## 注意事项

1. **填充值处理**：原始数据中的65535为填充值，定标后会转换为 `np.nan`

2. **内存使用**：完整图像大小为 5496×5496，每个波段约 115MB（float32），全波段处理时请注意内存限制

3. **定标方法选择**：
   - 科学分析建议使用查找表（LUT）方法，精度最高
   - 快速预览可使用线性方法，速度更快

4. **有效数据范围**：DN值通常小于4096，超出此范围会被裁剪到查找表有效索引范围

5. **并行处理**：批量处理时建议进程数不超过CPU核心数，避免资源竞争

---

## 依赖库

- `numpy` - 数值计算
- `h5py` - HDF文件读写
- `matplotlib` - 可视化（仅demo脚本需要）

---

## 参考资料

- FY-4B卫星AGRI传感器L1数据产品规格说明
- 国家卫星气象中心(NSMC)数据文档
