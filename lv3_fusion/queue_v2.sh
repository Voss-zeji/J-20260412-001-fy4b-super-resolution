#!/bin/bash
# 等待当前 run_deep_training 完成后，启动 v2 补跑剩余方法
set -e

REPO_DIR=~/jobs/J-20260412-001-fy4b-super-resolution
LOG=$REPO_DIR/lv3_fusion/run_deep_training_v2.log

echo "[$(date)] 等待当前训练进程结束..."

# 等待当前 run_deep_training 进程结束
while pgrep -f "run_deep_training.py" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] 当前训练已结束，启动 v2 补跑..."

cd $REPO_DIR
nohup /root/miniconda3/envs/mamba2/bin/python -u \
    lv3_fusion/run_deep_training.py \
    --epochs 200 \
    --band CH07 \
    > $LOG 2>&1 &

echo "[$(date)] v2 补跑已启动，PID: $!"
echo "日志: $LOG"
