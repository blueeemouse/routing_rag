#!/bin/bash
# DPO训练轮数搜索脚本 - MiniLM-L6-v2
# 搜索参数: epochs (基于已确定的最佳beta和lr)
#
# 用法:
#   bash scripts/search_param/search_dpo_minilm_epochs.sh --beta 0.01 --lr 1e-5 --gpu 3
#
# 说明:
#   - 在确定最佳beta和lr后，搜索最优训练轮数
#   - 防止过拟合或训练不足

# set -e  # 禁用严格模式，确保一个实验失败不会终止整个搜索

cd "$(dirname "$0")/../.."

# 默认使用搜索到的最佳配置
BETA=0.01
LEARNING_RATE=1e-5
BATCH_SIZE=16
GPU_ID=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --beta)
            BETA="$2"
            shift 2
            ;;
        --lr|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --beta 0.01 --lr 1e-5 [--gpu 3]"
            exit 1
            ;;
    esac
done

# 设置GPU
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "使用GPU: $GPU_ID"
fi

# 搜索空间 - 训练轮数
EPOCHS_LIST=(3 5 8 10 15)

# 固定参数
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
# 使用小数据集(5000条)加速超参数搜索
TRAIN_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_full.json"
VAL_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_val.json"
BASE_OUTPUT_DIR="router_models/dpo_search_minilm_epochs"

mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "DPO Epochs Search"
echo "Fixed: Beta=$BETA, LR=$LEARNING_RATE"
echo "Search: Epochs = ${EPOCHS_LIST[*]}"
echo "=========================================="

RESULTS_FILE="$BASE_OUTPUT_DIR/search_results_beta${BETA}_lr${LEARNING_RATE}.txt"
echo "Epochs Search Results (Beta=$BETA, LR=$LEARNING_RATE)" > "$RESULTS_FILE"
printf "%-10s %-20s %-15s\n" "Epochs" "Output Dir" "Best Val Acc" >> "$RESULTS_FILE"
echo "----------------------------------------" >> "$RESULTS_FILE"

TOTAL=${#EPOCHS_LIST[@]}
CURRENT=0

for EPOCHS in "${EPOCHS_LIST[@]}"; do
    CURRENT=$((CURRENT + 1))
    EXP_NAME="epochs${EPOCHS}_beta${BETA}_lr${LEARNING_RATE}"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXP_NAME"
    
    echo ""
    echo "[$CURRENT/$TOTAL] Training with epochs=$EPOCHS"
    
    if [[ -f "$OUTPUT_DIR/eval_result.json" ]]; then
        echo "实验已存在，跳过..."
    else
        # 确保输出目录存在
        mkdir -p "$OUTPUT_DIR"
        
        python router/train_dpo.py \
            --model_name "$MODEL_NAME" \
            --train_file "$TRAIN_FILE" \
            --val_file "$VAL_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --learning_rate $LEARNING_RATE \
            --beta $BETA \
            --max_length 512 \
            --logging_steps 50 \
            --eval_steps 100 \
            --save_total_limit 1 \
            --warmup_steps 100 \
            --fp16 \
            2>&1 | tee "$OUTPUT_DIR/training_console.log" || true
    fi
    
    # 读取结果
    if [[ -f "$OUTPUT_DIR/best_model_info.json" ]]; then
        BEST_ACC=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/best_model_info.json')); print(data.get('eval_accuracy', 'N/A'))" 2>/dev/null || echo "N/A")
    else
        BEST_ACC="N/A"
    fi
    
    printf "%-10s %-20s %-15s\n" "$EPOCHS" "$EXP_NAME" "$BEST_ACC" >> "$RESULTS_FILE"
    echo "  Result: $BEST_ACC"
done

echo ""
echo "=========================================="
echo "搜索完成！"
cat "$RESULTS_FILE"
echo "=========================================="
