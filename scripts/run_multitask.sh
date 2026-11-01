#!/bin/bash
# ==============================================================================
# Script tự động huấn luyện và đánh giá Multi-task Classification trên CIFAR-10
# Chuẩn giao thức mAP@54000 (1,000 query / 5,000 train / 54,000 gallery)
# ==============================================================================

set -e

PYTHON=".venv/bin/python"

echo "=========================================================="
echo " [1/4] Chạy HashNet 32-bit Multi-task Classification"
echo "=========================================================="
$PYTHON scripts/train.py --config configs/hashnet_32bit_multitask_54k.yaml
$PYTHON scripts/evaluate.py --config configs/hashnet_32bit_multitask_54k.yaml \
    --checkpoint outputs/experiments/hashnet_32bit_multitask_54k/checkpoint_final.pth

echo "=========================================================="
echo " [2/4] Chạy HashNet 16-bit Multi-task Classification"
echo "=========================================================="
$PYTHON scripts/train.py --config configs/hashnet_16bit_multitask_54k.yaml
$PYTHON scripts/evaluate.py --config configs/hashnet_16bit_multitask_54k.yaml \
    --checkpoint outputs/experiments/hashnet_16bit_multitask_54k/checkpoint_final.pth

echo "=========================================================="
echo " [3/4] Chạy DHN 16-bit Multi-task Classification"
echo "=========================================================="
$PYTHON scripts/train_dhn.py --config configs/dhn_16bit_multitask_54k.yaml
$PYTHON scripts/evaluate.py --config configs/dhn_16bit_multitask_54k.yaml \
    --checkpoint outputs/experiments/dhn_16bit_multitask_54k/checkpoint_final.pth

echo "=========================================================="
echo " [4/4] Chạy DHN 32-bit Multi-task Classification"
echo "=========================================================="
$PYTHON scripts/train_dhn.py --config configs/dhn_32bit_multitask_54k.yaml
$PYTHON scripts/evaluate.py --config configs/dhn_32bit_multitask_54k.yaml \
    --checkpoint outputs/experiments/dhn_32bit_multitask_54k/checkpoint_final.pth

echo "=========================================================="
echo " HOÀN TẤT TOÀN BỘ EXPERIMENTS MULTI-TASK!"
echo "=========================================================="

