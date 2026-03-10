#!/bin/bash
# DPO Router 训练启动脚本
# 
# 用法:
#   bash scripts/train/train_dpo_router.sh                    # 使用所有GPU
#   bash scripts/train/train_dpo_router.sh --gpu 3            # 只使用3号GPU
#   bash scripts/train/train_dpo_router.sh --epochs 5 --batch_size 8
#
# 参数:
#   --gpu: 指定GPU卡号（如0,1,2,3），默认使用所有可用GPU
#
# 功能:
#   - 使用 DPO (Direct Preference Optimization) 训练路由器
#   - 自动保存验证集准确率最好的模型
#   - 模型保存到 router_models/dpo_router/ 目录

set -e

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "DPO Router Training"
echo "=========================================="
echo ""

# 默认配置
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
TRAIN_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_full.json"
VAL_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_val.json"
OUTPUT_DIR="router_models/dpo_router"

BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=1e-5
BETA=0.1
MAX_LENGTH=512
GPU_ID=""  # 默认不指定，使用所有可用GPU

# 解析额外参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置GPU（如果指定了）
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "使用GPU: $GPU_ID"
else
    echo "使用所有可用GPU"
fi

echo "配置信息:"
echo "  模型: $MODEL_NAME"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  DPO beta: $BETA"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 启动训练
python router/train_dpo.py \
    --model_name "$MODEL_NAME" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --beta $BETA \
    --max_length $MAX_LENGTH \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --warmup_steps 100 \
    --fp16

echo ""
echo "=========================================="
echo "训练完成!"
echo "模型保存到: $OUTPUT_DIR"
echo "最佳模型: $OUTPUT_DIR/checkpoint-XXX (验证集准确率最高)"
echo "=========================================="
