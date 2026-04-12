#!/bin/bash
# 远程GPU服务器测试脚本
# 每种方法运行30分钟后自动中止

cd ~/jobs/J-20260412-001-fy4b-super-resolution || exit 1

# 创建结果目录
mkdir -p results/lv1_macro_ch07

# 方法列表
METHODS=(
    "01_baseline_bicubic"
    "02_baseline_srcnn"
    "03_method_edsr"
    "04_method_pftsr"
    "05_method_swinir"
    "06_method_tinynina"
    "07_method_m2ir"
    "08_method_realrestorer"
    "09_method_lcmsr"
)

# 运行每种方法，限时30分钟
echo "======================================"
echo "开始CH07通道9种方法测试 (各30分钟)"
echo "开始时间: $(date)"
echo "======================================"

for method in "${METHODS[@]}"; do
    echo ""
    echo "--------------------------------------"
    echo "运行方法: $method"
    echo "开始时间: $(date)"
    echo "--------------------------------------"

    # 创建日志文件
    LOG_FILE="results/lv1_macro_ch07/${method}_$(date +%Y%m%d_%H%M%S).log"
    RESULT_FILE="results/lv1_macro_ch07/${method}_result.json"

    # 记录开始信息
    echo "{" > "$RESULT_FILE"
    echo "  \"method\": \"$method\"," >> "$RESULT_FILE"
    echo "  \"band\": \"CH07\"," >> "$RESULT_FILE"
    echo "  \"start_time\": \"$(date -Iseconds)\"," >> "$RESULT_FILE"
    echo "  \"max_duration_minutes\": 30," >> "$RESULT_FILE"

    # 运行方法，限时30分钟 (1800秒)
    timeout 1800 python lv1_macro/methods/${method}/main.py \
        --band CH07 \
        --epochs 1000 \
        --batch-size 16 \
        --lr 0.0001 \
        --output "$RESULT_FILE.tmp" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    # 记录结束信息
    echo "  \"end_time\": \"$(date -Iseconds)\"," >> "$RESULT_FILE"
    echo "  \"exit_code\": $EXIT_CODE," >> "$RESULT_FILE"

    # 检查是否超时
    if [ $EXIT_CODE -eq 124 ]; then
        echo "  \"status\": \"timeout\"," >> "$RESULT_FILE"
        echo "  \"note\": \"运行30分钟后超时中止\"" >> "$RESULT_FILE"
        echo "⚠️  $method 已超时 (30分钟)"
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "  \"status\": \"completed\"," >> "$RESULT_FILE"
        echo "  \"note\": \"正常完成\"" >> "$RESULT_FILE"
        echo "✅ $method 正常完成"
    else
        echo "  \"status\": \"error\"," >> "$RESULT_FILE"
        echo "  \"note\": \"退出码 $EXIT_CODE\"" >> "$RESULT_FILE"
        echo "❌ $method 出错 (退出码 $EXIT_CODE)"
    fi

    echo "}" >> "$RESULT_FILE"

    # 合并临时结果（如果存在）
    if [ -f "$RESULT_FILE.tmp" ]; then
        # 提取有效JSON内容
        python3 << EOF
import json
import sys

try:
    with open("$RESULT_FILE.tmp", 'r') as f:
        data = json.load(f)
    with open("$RESULT_FILE", 'r') as f:
        meta = json.load(f)

    # 合并数据
    merged = {**meta, **data}

    with open("$RESULT_FILE", 'w') as f:
        json.dump(merged, f, indent=2)
    print("结果已合并")
except Exception as e:
    print(f"合并失败: {e}")
EOF
        rm -f "$RESULT_FILE.tmp"
    fi

    echo "结束时间: $(date)"
    echo "日志文件: $LOG_FILE"
    echo "结果文件: $RESULT_FILE"
    echo ""

done

echo "======================================"
echo "所有方法测试完成"
echo "结束时间: $(date)"
echo "结果目录: results/lv1_macro_ch07/"
echo "======================================"

# 生成汇总报告
echo "生成汇总报告..."
python3 << 'EOF'
import json
import os
from pathlib import Path

results_dir = Path("results/lv1_macro_ch07")
summary = []

for result_file in sorted(results_dir.glob("*_result.json")):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)

        summary.append({
            "method": data.get("method", "unknown"),
            "status": data.get("status", "unknown"),
            "val_psnr": data.get("val_psnr", None),
            "val_ssim": data.get("val_ssim", None),
            "model_params": data.get("model_params", None),
            "train_epochs": data.get("train_epochs", None),
            "train_time": data.get("train_time", None),
        })
    except Exception as e:
        print(f"读取 {result_file} 失败: {e}")

# 保存汇总
with open("results/lv1_macro_ch07/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# 打印表格
print("\n" + "="*80)
print("CH07通道测试结果汇总")
print("="*80)
print(f"{'方法':<25} {'状态':<12} {'PSNR':<8} {'SSIM':<8} {'参数量':<12} {'轮数':<6}")
print("-"*80)
for item in summary:
    psnr = f"{item['val_psnr']:.2f}" if item['val_psnr'] else "N/A"
    ssim = f"{item['val_ssim']:.4f}" if item['val_ssim'] else "N/A"
    params = f"{item['model_params']:,}" if item['model_params'] else "N/A"
    epochs = str(item['train_epochs']) if item['train_epochs'] else "N/A"
    print(f"{item['method']:<25} {item['status']:<12} {psnr:<8} {ssim:<8} {params:<12} {epochs:<6}")
print("="*80)
EOF

echo ""
echo "汇总报告: results/lv1_macro_ch07/summary.json"
