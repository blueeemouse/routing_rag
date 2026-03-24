#!/bin/bash
# Hybrid Representation Fusion Router Training Script
# 混合表征融合路由器训练脚本
#
# 【特点】LLM 内部表征 + MiniLM 语义表征 + Cross Attention 融合
#
# 架构:
#   内部表征 (2048-d) ──┐
#                      │ Cross Attention
#   语义表征 (384-d) ──┘
#                      │
#                  Classifier
#                      │
#               路由决策 (no_rag/naive_rag)
#
# 使用方法:
#   ./train_hybrid_representation_fusion.sh                    # 正常训练
#   ./train_hybrid_representation_fusion.sh --quick-test       # 快速测试 (10 steps)
#   ./train_hybrid_representation_fusion.sh --bidirectional    # 使用双向 Cross Attention
#   ./train_hybrid_representation_fusion.sh --freeze-backbone  # 冻结 MiniLM backbone

# 默认参数
CONFIG="config/train_hybrid_representation_fusion.yaml"
OUTPUT_DIR="router_models/hybrid_representation_fusion_5000_concat_all_mean"
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=1e-4
EVAL_STEPS=50
SEED=42
QUICK_TEST=false
FUSION_TYPE="cross_attn"
FREEZE_BACKBONE=false
FREEZE_INTERNAL_PROJ=false
RESUME=""

# 数据路径
REPRESENTATION_DIR="outputs/representations/fp16_qwen2.5-3b-instruct_train5000"
LABELS_PATH="HotpotQA_train_data/label_analysis/all_labels_vllm_qwen_with_tie_converted.json"
# REPRESENTATION_TYPE="deep_last_token"
REPRESENTATION_TYPE="concat_all_mean"

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
        --representation-dir)
            REPRESENTATION_DIR="$2"
            shift 2
            ;;
        --labels-path)
            LABELS_PATH="$2"
            shift 2
            ;;
        --representation-type)
            REPRESENTATION_TYPE="$2"
            shift 2
            ;;
        --bidirectional)
            FUSION_TYPE="bidirectional_cross_attn"
            shift
            ;;
        --freeze-backbone)
            FREEZE_BACKBONE=true
            shift
            ;;
        --freeze-internal-proj)
            FREEZE_INTERNAL_PROJ=true
            shift
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
            echo "Usage: ./train_hybrid_representation_fusion.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config <path>              Config file path"
            echo "  --output-dir <path>          Output directory"
            echo "  --epochs <int>               Number of epochs (default: 10)"
            echo "  --batch-size <int>           Batch size (default: 32)"
            echo "  --learning-rate <float>      Learning rate (default: 0.0001)"
            echo "  --eval-steps <int>           Evaluation steps (default: 100)"
            echo "  --seed <int>                 Random seed (default: 42)"
            echo "  --representation-dir <path>  Internal representation directory"
            echo "  --labels-path <path>         Labels JSON file path"
            echo "  --representation-type <type> Representation type (default: deep_last_token)"
            echo "  --bidirectional              Use bidirectional cross attention"
            echo "  --freeze-backbone            Freeze MiniLM backbone"
            echo "  --freeze-internal-proj       Freeze internal representation projection layer"
            echo "  --resume <path>              Resume from checkpoint"
            echo "  --quick-test                 Quick test mode (10 steps)"
            echo "  -h, --help                   Show this help message"
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
    OUTPUT_DIR="router_models/hybrid_rep_fusion_${TIMESTAMP}"
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
echo_cyan "混合表征融合路由器训练"
echo_cyan "========================================="
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "数据配置:"
echo "  表征目录: $REPRESENTATION_DIR"
echo "  标签文件: $LABELS_PATH"
echo "  表征类型: $REPRESENTATION_TYPE"
echo ""
echo "模型配置:"
echo "  融合方式: $FUSION_TYPE"
echo "  冻结 Backbone: $FREEZE_BACKBONE"
echo "  冻结内部表征投影: $FREEZE_INTERNAL_PROJ"
echo ""
echo "训练配置:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Eval Steps: $EVAL_STEPS"
echo "  Seed: $SEED"
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
COMMAND="$COMMAND --representation_dir $REPRESENTATION_DIR"
COMMAND="$COMMAND --labels_path $LABELS_PATH"
COMMAND="$COMMAND --representation_type $REPRESENTATION_TYPE"
COMMAND="$COMMAND --fusion_type $FUSION_TYPE"

if [[ "$FREEZE_BACKBONE" == true ]]; then
    COMMAND="$COMMAND --freeze_backbone"
fi

if [[ "$FREEZE_INTERNAL_PROJ" == true ]]; then
    COMMAND="$COMMAND --freeze_internal_rep_proj"
fi

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
echo ""
eval $COMMAND

echo ""
echo_green "训练完成!"
echo_green "模型保存路径: $OUTPUT_DIR"
