#!/bin/bash
# lv2_micro 自动化实验脚本
# 使用方法: ./run_experiment.sh <band> <experiment_name> [description]

set -e

# 参数
BAND=${1:-CH07}
EXPERIMENT_NAME=${2:-$(date +%Y%m%d)_exp}
DESCRIPTION=${3:-"experiment"}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "FY4B Super-Resolution Experiment"
echo "========================================"
echo "Band: $BAND"
echo "Experiment: $EXPERIMENT_NAME"
echo "Description: $DESCRIPTION"
echo "========================================"

# 检查实验目录是否存在
if [ ! -d "experiments/$EXPERIMENT_NAME" ]; then
    echo -e "${RED}Error: experiments/$EXPERIMENT_NAME not found${NC}"
    echo "请先创建实验目录: cp -r <baseline> experiments/$EXPERIMENT_NAME"
    exit 1
fi

# 运行实验
echo -e "\n[1/3] Running experiment..."
python experiments/$EXPERIMENT_NAME/main.py \
  --band $BAND \
  --output experiments/$EXPERIMENT_NAME/result.json \
  2>&1 | tee experiments/$EXPERIMENT_NAME/run.log

# 检查结果
if [ ! -f "experiments/$EXPERIMENT_NAME/result.json" ]; then
    echo -e "${RED}Error: Experiment failed - result.json not found${NC}"
    exit 1
fi

# 提取结果
echo -e "\n[2/3] Extracting results..."
PSNR=$(grep '"val_psnr"' experiments/$EXPERIMENT_NAME/result.json | grep -o '[0-9.]*' | head -1)
MEMORY=$(grep '"memory_gb"' experiments/$EXPERIMENT_NAME/result.json | grep -o '[0-9.]*' | head -1 || echo "4.5")

echo "val_psnr: $PSNR dB"
echo "memory_gb: $MEMORY GB"

# 获取当前最佳结果
echo -e "\n[3/3] Comparing with best result..."
if [ -f "results.tsv" ] && [ $(wc -l < results.tsv) -gt 1 ]; then
    BEST_PSNR=$(tail -n +2 results.tsv | sort -k2 -nr | head -1 | cut -f2)
    BEST_EXP=$(tail -n +2 results.tsv | sort -k2 -nr | head -1 | cut -f1)

    echo "Current best: $BEST_EXP ($BEST_PSNR dB)"

    # 比较结果
    if (( $(echo "$PSNR > $BEST_PSNR" | bc -l) )); then
        STATUS="keep"
        echo -e "${GREEN}✓ Improved: $PSNR > $BEST_PSNR dB${NC}"
    else
        STATUS="discard"
        echo -e "${YELLOW}✗ Not improved: $PSNR <= $BEST_PSNR dB${NC}"
    fi
else
    STATUS="keep"
    echo -e "${GREEN}✓ First experiment, set as baseline${NC}"
fi

# 记录结果
echo -e "$EXPERIMENT_NAME\t$PSNR\t$MEMORY\t$STATUS\t$DESCRIPTION" >> results.tsv
echo -e "\nRecorded to results.tsv: $STATUS"

echo "========================================"
echo -e "Experiment ${GREEN}$EXPERIMENT_NAME${NC} completed: $STATUS"
echo "========================================"
