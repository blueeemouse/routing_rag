#!/bin/bash
# 批量训练不同内部表征的路由器
# 用法: ./scripts/train/batch_train_representations.sh

# 配置
BATCH_SIZE=64
EPOCHS=10
EVAL_STEPS=50
LEARNING_RATE=1e-4

# 训练数据目录
TRAIN_DATA="outputs/representations/fp16_qwen2.5-3b-instruct_train5000"
VAL_DATA="outputs/representations/fp16_qwen2.5-3b-instruct_test1000"

# 定义表征类型及其维度
declare -A REPR_DIMS=(
    ["shallow_mean"]=2048
    ["shallow_last_token"]=2048
    ["middle_mean"]=2048
    ["middle_last_token"]=2048
    ["deep_mean"]=2048
    ["deep_last_token"]=2048
    ["concat_deep"]=4096
    ["concat_all_mean"]=6144
    ["concat_all_last"]=6144
    ["concat_all"]=12288
)

# 创建输出目录
OUTPUT_BASE="outputs/internal_rep_router_experiments_5000_no_0.8_naive_1.2"
mkdir -p "$OUTPUT_BASE"

# 日志文件
LOG_FILE="$OUTPUT_BASE/batch_train_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "批量训练开始: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 遍历每种表征类型
for REPR_TYPE in "${!REPR_DIMS[@]}"; do
    DIM=${REPR_DIMS[$REPR_TYPE]}
    OUTPUT_DIR="$OUTPUT_BASE/${REPR_TYPE}"
    
    echo "" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "训练表征类型: $REPR_TYPE (维度: $DIM)" | tee -a "$LOG_FILE"
    echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    python router/train_router.py \
        --config config/train_internal_representation.yaml \
        --data_source internal_representation \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --eval_steps $EVAL_STEPS \
        --learning_rate $LEARNING_RATE \
        --output_dir "$OUTPUT_DIR" \
        --representation_type "$REPR_TYPE" \
        --representation_dim $DIM \
        --class_weights "no_rag=0.8,naive_rag=1.2"
        2>&1 | tee -a "$LOG_FILE"
    
    echo "完成: $REPR_TYPE" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "批量训练完成: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 汇总结果
echo "" | tee -a "$LOG_FILE"
echo "结果汇总:" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"
for REPR_TYPE in "${!REPR_DIMS[@]}"; do
    RESULT_FILE="$OUTPUT_BASE/${REPR_TYPE}/checkpoint_best_val/train_state.json"
    if [ -f "$RESULT_FILE" ]; then
        BEST_ACC=$(grep -o '"best_val_accuracy": [0-9.]*' "$RESULT_FILE" | grep -o '[0-9.]*$')
        echo "$REPR_TYPE: best_val_accuracy = $BEST_ACC" | tee -a "$LOG_FILE"
    else
        echo "$REPR_TYPE: 未找到结果" | tee -a "$LOG_FILE"
    fi
done
