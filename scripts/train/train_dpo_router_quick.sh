#!/bin/bash
# DPO Router 快速测试脚本
# 
# 用法:
#   bash scripts/train/train_dpo_router_quick.sh
#
# 功能:
#   - 使用少量数据快速验证DPO训练流程
#   - 1个epoch，小batch，快速验证代码正确性

set -e

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "DPO Router Quick Test"
echo "=========================================="
echo ""

OUTPUT_DIR="router_models/dpo_router_quick_test"

# 清理旧的测试输出
rm -rf "$OUTPUT_DIR"

echo "快速测试配置:"
echo "  模型: sentence-transformers/all-MiniLM-L6-v2"
echo "  批次大小: 4"
echo "  训练轮数: 1"
echo "  学习率: 1e-5"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 启动快速训练测试
python router/train_dpo.py \
    --model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --train_file "HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_full.json" \
    --val_file "HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_val.json" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --epochs 1 \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --max_length 512 \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 1 \
    --warmup_steps 50 \
    --fp16

echo ""
echo "=========================================="
echo "快速测试完成!"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
