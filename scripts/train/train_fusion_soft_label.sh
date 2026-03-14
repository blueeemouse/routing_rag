#!/bin/bash
# Fusion Soft Label Router Training Script
# 融合模型软标签路由器训练脚本
#
# 【特点】语义特征 + 统计特征 + 软标签训练
#
# 训练配置：
# - 模型: GatedFusionModel (语义 + 统计特征融合)
# - 训练器: FusionSoftLabelTrainer (CrossEntropyLoss 支持软标签)
# - 数据集: FusionSoftLabelDataset (返回 input_ids, attention_mask, soft_label向量)
# - 数据: all_labels_soft.json (包含 soft_label 字段)
#
# 使用方法:
#   ./train_fusion_soft_label.sh                    # 正常训练
#   ./train_fusion_soft_label.sh --quick-test       # 快速测试 (10 steps)
#   ./train_fusion_soft_label.sh --epochs 10        # 自定义 epochs

# 默认参数
CONFIG="config/train_fusion_soft_label.yaml"
OUTPUT_DIR="router_models/fusion_soft_label_5000"
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=7e-5
EVAL_STEPS=50
SEED=42
QUICK_TEST=false
RESUME=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --eval-steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./train_fusion_soft_label.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config <path>         Config file path (default: config/train_fusion_soft_label.yaml)"
            echo "  --output-dir <path>     Output directory (default: router_models/fusion_soft_label_<timestamp>)"
            echo "  --epochs <int>          Number of epochs (default: 5)"
            echo "  --batch-size <int>      Batch size (default: 16)"
            echo "  --learning-rate <float> Learning rate (default: 0.00005)"
            echo "  --eval-steps <int>      Evaluation steps (default: 100)"
            echo "  --seed <int>            Random seed (default: 42)"
            echo "  --resume <path>         Resume from checkpoint"
            echo "  --quick-test            Quick test mode (10 steps)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 输出目录
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="router_models/fusion_soft_label_${TIMESTAMP}"
fi

# 颜色输出函数
echo_cyan() {
    echo -e "\033[36m$1\033[0m"
}

echo_green() {
    echo -e "\033[32m$1\033[0m"
}

echo_yellow() {
    echo -e "\033[33m$1\033[0m"
}

echo_gray() {
    echo -e "\033[90m$1\033[0m"
}

echo_cyan "========================================="
echo_cyan "融合模型软标签路由器训练"
echo_cyan "========================================="
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Eval Steps: $EVAL_STEPS"
echo "Seed: $SEED"
echo_cyan "========================================="

# 构建命令
COMMAND="python router/train_router.py"
COMMAND="$COMMAND --config $CONFIG"
COMMAND="$COMMAND --output_dir $OUTPUT_DIR"
COMMAND="$COMMAND --epochs $EPOCHS"
COMMAND="$COMMAND --batch_size $BATCH_SIZE"
COMMAND="$COMMAND --learning_rate $LEARNING_RATE"
COMMAND="$COMMAND --eval_steps $EVAL_STEPS"
COMMAND="$COMMAND --seed $SEED"

if [[ -n "$RESUME" ]]; then
    COMMAND="$COMMAND --resume $RESUME"
fi

# 快速测试模式
if [[ "$QUICK_TEST" == true ]]; then
    COMMAND="$COMMAND --max_steps 10 --overfit_single_batch"
    echo ""
    echo_yellow "快速测试模式 (10 steps)..."
else
    echo ""
    echo_green "开始训练..."
fi

# 执行训练
echo_gray "命令: $COMMAND"
eval $COMMAND

echo ""
echo_green "训练完成!"
echo_green "模型保存路径: $OUTPUT_DIR"
