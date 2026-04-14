#!/bin/bash
cd /root/jobs/J-20260412-001-fy4b-super-resolution
PYTHON=/root/miniconda3/envs/mamba2/bin/python
OUTDIR=/root/jobs/J-20260412-001-fy4b-super-resolution/test_results
mkdir -p $OUTDIR

for method in 02_baseline_srcnn 03_method_edsr 05_method_swinir 07_method_m2ir; do
    echo "===== Testing $method ====="
    $PYTHON lv1_macro/methods/$method/main.py --band CH07 --epochs 2 --batch-size 16 --output $OUTDIR/$method.json 2>&1 | tail -20
    echo ""
done
