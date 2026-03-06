#!/bin/bash
# 测试加权数据集的训练功能
# 使用小批量数据快速验证

# 配置
CONFIG="config/train_classification_5000.yaml"
TRAIN_DATA="HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json"
VAL_DATA="evaluation_results/router_test_labels.json"
TIE_WEIGHT=0.5

echo "========================================"
echo "测试加权数据集训练"
echo "========================================"
echo "配置文件: $CONFIG"
echo "训练数据: $TRAIN_DATA"
echo "Tie样本权重: $TIE_WEIGHT"
echo ""

# 运行训练
python router/train_router.py \
    --config "$CONFIG" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --tie_weight $TIE_WEIGHT \
    --epochs 1 \
    --max_steps 10

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
