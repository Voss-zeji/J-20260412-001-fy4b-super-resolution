#!/usr/bin/env python3
"""
测试Calibration后的数据正确性
验证点：
1. 文件数量和命名匹配
2. HDF文件结构正确
3. 数据范围和统计信息合理
4. 2000M和4000M文件一一对应
"""

import os
import sys
import glob
import re
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import h5py
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("警告: 未安装h5py或numpy，将只进行文件结构检查")


class CalibrationDataTester:
    """Calibration数据测试器"""

    def __init__(self, data_root="/root/autodl-tmp/FY-4B/calibration"):
        self.data_root = data_root
        self.results = []
        self.errors = []

    def log(self, msg, level="INFO"):
        """记录日志"""
        prefix = {"INFO": "[✓]", "WARN": "[!]", "ERROR": "[✗]"}.get(level, "[*]")
        print(f"{prefix} {msg}")
        self.results.append((level, msg))

    def test_directory_structure(self):
        """测试目录结构"""
        print("\n" + "="*60)
        print("测试1: 目录结构检查")
        print("="*60)

        expected_dirs = [
            "2000M/CH07", "2000M/CH08",
            "4000M/CH07", "4000M/CH08"
        ]

        all_exist = True
        for dir_path in expected_dirs:
            full_path = os.path.join(self.data_root, dir_path)
            if os.path.exists(full_path):
                self.log(f"目录存在: {dir_path}")
            else:
                self.log(f"目录缺失: {dir_path}", "ERROR")
                all_exist = False

        return all_exist

    def test_file_counts(self):
        """测试文件数量"""
        print("\n" + "="*60)
        print("测试2: 文件数量检查")
        print("="*60)

        channels = ["CH07", "CH08"]
        resolutions = ["2000M", "4000M"]

        stats = {}
        for res in resolutions:
            stats[res] = {}
            for ch in channels:
                dir_path = os.path.join(self.data_root, res, ch)
                if os.path.exists(dir_path):
                    files = glob.glob(os.path.join(dir_path, "*.HDF"))
                    stats[res][ch] = len(files)
                    self.log(f"{res}/{ch}: {len(files)} 个文件")
                else:
                    stats[res][ch] = 0
                    self.log(f"{res}/{ch}: 目录不存在", "ERROR")

        # 检查数量匹配
        for ch in channels:
            if stats["2000M"].get(ch, 0) != stats["4000M"].get(ch, 0):
                self.log(f"{ch}: 2000M和4000M文件数量不匹配!", "WARN")
            else:
                self.log(f"{ch}: 2000M和4000M文件数量匹配 ✓")

        return stats

    def test_file_naming(self):
        """测试文件命名规范"""
        print("\n" + "="*60)
        print("测试3: 文件命名规范检查")
        print("="*60)

        # 期望格式: FY4B_CH07_CAL_20250301000000.HDF
        pattern = re.compile(r'FY4B_CH\d{2}_CAL_\d{14}\.HDF')

        all_valid = True
        for res in ["2000M", "4000M"]:
            for ch in ["CH07", "CH08"]:
                dir_path = os.path.join(self.data_root, res, ch)
                if not os.path.exists(dir_path):
                    continue

                files = glob.glob(os.path.join(dir_path, "*.HDF"))
                for f in files:
                    basename = os.path.basename(f)
                    if pattern.match(basename):
                        continue
                    else:
                        self.log(f"命名不规范: {basename}", "WARN")
                        all_valid = False

        if all_valid:
            self.log("所有文件命名符合规范: FY4B_CHxx_CAL_YYYYMMDDHHMMSS.HDF")

        return all_valid

    def test_hdf_structure(self, sample_size=3):
        """测试HDF文件结构"""
        if not HAS_DEPS:
            self.log("跳过HDF结构测试 (缺少依赖)", "WARN")
            return False

        print("\n" + "="*60)
        print("测试4: HDF文件结构检查")
        print("="*60)

        for res in ["2000M", "4000M"]:
            for ch in ["CH07", "CH08"]:
                dir_path = os.path.join(self.data_root, res, ch)
                if not os.path.exists(dir_path):
                    continue

                files = sorted(glob.glob(os.path.join(dir_path, "*.HDF")))
                if not files:
                    continue

                # 抽取样本检查
                samples = files[:min(sample_size, len(files))]

                for sample in samples:
                    try:
                        with h5py.File(sample, 'r') as f:
                            # 检查数据集存在 (实际使用CH07而不是Channel07)
                            channel_key = ch  # 直接使用CH07/CH08
                            if channel_key not in f:
                                self.log(f"{os.path.basename(sample)}: 缺少{channel_key}数据集", "ERROR")
                                continue

                            data = f[channel_key]

                            # 检查基本属性
                            required_attrs = ['band_name', 'wavelength', 'type', 'unit']
                            missing_attrs = [a for a in required_attrs if a not in data.attrs]

                            if missing_attrs:
                                self.log(f"{os.path.basename(sample)}: 缺少属性 {missing_attrs}", "WARN")
                            else:
                                self.log(f"{os.path.basename(sample)}: 结构正确 ✓")

                            # 检查数据形状
                            expected_shape = (5496, 5496) if res == "2000M" else (2748, 2748)
                            if data.shape != expected_shape:
                                self.log(f"  警告: 形状{data.shape}，期望{expected_shape}", "WARN")
                            else:
                                self.log(f"  形状: {data.shape} ✓")

                    except Exception as e:
                        self.log(f"{os.path.basename(sample)}: 读取失败 - {e}", "ERROR")

        return True

    def test_data_statistics(self, sample_size=3):
        """测试数据统计信息"""
        if not HAS_DEPS:
            self.log("跳过数据统计测试 (缺少依赖)", "WARN")
            return False

        print("\n" + "="*60)
        print("测试5: 数据统计信息检查")
        print("="*60)

        for res in ["2000M", "4000M"]:
            for ch in ["CH07", "CH08"]:
                dir_path = os.path.join(self.data_root, res, ch)
                if not os.path.exists(dir_path):
                    continue

                files = sorted(glob.glob(os.path.join(dir_path, "*.HDF")))
                if not files:
                    continue

                samples = files[:min(sample_size, len(files))]
                channel_key = ch  # 直接使用CH07/CH08

                for sample in samples:
                    try:
                        with h5py.File(sample, 'r') as f:
                            data = f[channel_key][()]

                            # 检查NaN比例
                            nan_ratio = np.sum(np.isnan(data)) / data.size * 100
                            self.log(f"{os.path.basename(sample)}: NaN比例={nan_ratio:.2f}%")

                            # 检查数值范围 (亮温应在合理范围)
                            valid_data = data[~np.isnan(data)]
                            if len(valid_data) > 0:
                                min_val, max_val = valid_data.min(), valid_data.max()
                                self.log(f"  数值范围: {min_val:.2f} K ~ {max_val:.2f} K")

                                # 亮温合理性检查 (150K-400K)
                                if min_val < 150 or max_val > 400:
                                    self.log(f"  警告: 数值超出预期范围(150-400K)", "WARN")

                    except Exception as e:
                        self.log(f"{os.path.basename(sample)}: 统计失败 - {e}", "ERROR")

        return True

    def test_timestamp_matching(self):
        """测试时间戳匹配"""
        if not HAS_DEPS:
            self.log("跳过时间戳匹配测试 (缺少依赖)", "WARN")
            return False

        print("\n" + "="*60)
        print("测试6: 2000M和4000M文件时间戳匹配检查")
        print("="*60)

        pattern = re.compile(r'FY4B_CH\d{2}_CAL_(\d{14})\.HDF')

        for ch in ["CH07", "CH08"]:
            dir_2000m = os.path.join(self.data_root, "2000M", ch)
            dir_4000m = os.path.join(self.data_root, "4000M", ch)

            if not os.path.exists(dir_2000m) or not os.path.exists(dir_4000m):
                continue

            files_2000m = {pattern.search(f).group(1) for f in os.listdir(dir_2000m)
                          if pattern.search(f)}
            files_4000m = {pattern.search(f).group(1) for f in os.listdir(dir_4000m)
                          if pattern.search(f)}

            common = files_2000m & files_4000m
            only_2000m = files_2000m - files_4000m
            only_4000m = files_4000m - files_2000m

            self.log(f"{ch}: 共同文件={len(common)}, 仅2000M={len(only_2000m)}, 仅4000M={len(only_4000m)}")

            if only_2000m:
                self.log(f"  仅2000M的前5个: {list(only_2000m)[:5]}", "WARN")
            if only_4000m:
                self.log(f"  仅4000M的前5个: {list(only_4000m)[:5]}", "WARN")

        return True

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*70)
        print("FY-4B Calibration 数据测试")
        print(f"数据根目录: {self.data_root}")
        print("="*70)

        # 运行测试
        self.test_directory_structure()
        self.test_file_counts()
        self.test_file_naming()
        self.test_hdf_structure()
        self.test_data_statistics()
        self.test_timestamp_matching()

        # 总结
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)

        info_count = sum(1 for r in self.results if r[0] == "INFO")
        warn_count = sum(1 for r in self.results if r[0] == "WARN")
        error_count = sum(1 for r in self.results if r[0] == "ERROR")

        print(f"通过: {info_count} | 警告: {warn_count} | 错误: {error_count}")

        if error_count == 0:
            print("\n✓ 所有关键测试通过!")
            return True
        else:
            print(f"\n✗ 发现 {error_count} 个错误，请检查数据!")
            return False


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        # 尝试常见路径
        candidates = [
            "/root/autodl-tmp/FY-4B/calibration",
            "/root/autodl-tmp/Calibration-FY4B",
            "./calibration",
        ]
        data_root = None
        for c in candidates:
            if os.path.exists(c):
                data_root = c
                break

        if data_root is None:
            print("错误: 未找到数据目录，请指定路径:")
            print(f"  python {sys.argv[0]} /path/to/calibration")
            sys.exit(1)

    tester = CalibrationDataTester(data_root)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
