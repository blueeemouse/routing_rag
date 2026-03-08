#!/bin/bash
# Tie权重搜索脚本
# 测试不同的tie样本权重，观察训练效果

# 输出目录
EXPERIMENTS_ROOT="router_models/tie_weight_search"
LOG_FILE="$EXPERIMENTS_ROOT/tie_weight_search_log.txt"

# 数据配置
TRAIN_DATA="HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json"
VAL_DATA="evaluation_results/router_test_labels.json"

# 搜索的tie权重
TIE_WEIGHTS=(0.2 0.5 1.0 1.5 2.0)

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "Tie权重搜索脚本"
echo "========================================"
echo ""
echo "配置:"
echo "  Backbone 1: all-MiniLM-L6-v2 (dc, lr=2e-4, wd=0.01)"
echo "  Backbone 2: bge-base-en-v1.5 (feature_fused, lr=7e-5, wd=10)"
echo "  Temperature: 0.5"
echo "  Epochs: 5"
echo "  Tie权重搜索: ${TIE_WEIGHTS[@]}"
echo "  输出目录: $EXPERIMENTS_ROOT"
echo "  GPU: 0"
echo ""

# 创建输出目录
mkdir -p "$EXPERIMENTS_ROOT"

# 初始化日志
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Tie权重搜索开始" > "$LOG_FILE"
echo "配置:" >> "$LOG_FILE"
echo "  Backbone: MiniLM (dc) + BGE (feature_fused)" >> "$LOG_FILE"
echo "  Temperature: 0.5" >> "$LOG_FILE"
echo "  Epochs: 5" >> "$LOG_FILE"
echo "  Tie权重: ${TIE_WEIGHTS[@]}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 统计变量
TOTAL=$((${#TIE_WEIGHTS[@]} * 2))  # 2种backbone
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

echo "准备运行 $TOTAL 个实验 (2 backbones × ${#TIE_WEIGHTS[@]} tie权重)..."
echo ""

# ============================================
# Backbone 1: all-MiniLM-L6-v2 (dc model)
# ============================================
BACKBONE_NAME="all-MiniLM-L6-v2"
BACKBONE="sentence-transformers/all-MiniLM-L6-v2"
MODEL_TYPE="dc"
TRAINER_TYPE="classification"
HIDDEN_SIZE="384"
LEARNING_RATE="2e-4"
TEMPERATURE="0.5"
EPOCHS="5"
WEIGHT_DECAY="0.01"

echo "----------------------------------------"
echo "Backbone 1: $BACKBONE_NAME (model_type=$MODEL_TYPE, lr=$LEARNING_RATE, wd=$WEIGHT_DECAY)"
echo "----------------------------------------"
echo ""

for tie_weight in "${TIE_WEIGHTS[@]}"; do
    # 生成输出目录名
    OUTPUT_DIR="$EXPERIMENTS_ROOT/${BACKBONE_NAME}/tie_weight_${tie_weight}"
    
    # 构建参数
    PARAMS=(
        "router/train_router.py"
        "--model_type" "$MODEL_TYPE"
        "--trainer_type" "$TRAINER_TYPE"
        "--backbone" "$BACKBONE"
        "--train_data" "$TRAIN_DATA"
        "--val_data" "$VAL_DATA"
        "--learning_rate" "$LEARNING_RATE"
        "--temperature" "$TEMPERATURE"
        "--epochs" "$EPOCHS"
        "--weight_decay" "$WEIGHT_DECAY"
        "--tie_weight" "$tie_weight"
        "--output_dir" "$OUTPUT_DIR"
    )
    
    # 显示当前实验信息
    PROGRESS="[$(($COMPLETED + $FAILED + 1))/$TOTAL] $BACKBONE_NAME, tie_weight=$tie_weight"
    echo "$PROGRESS"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始: $BACKBONE_NAME, tie_weight=$tie_weight" >> "$LOG_FILE"
    
    # 执行训练
    echo "命令: python ${PARAMS[@]}"
    python "${PARAMS[@]}"
    EXIT_CODE=$?
    
    # 检查结果
    if [ $EXIT_CODE -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "✓ 实验完成: $BACKBONE_NAME, tie_weight=$tie_weight"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [成功] 实验完成: $BACKBONE_NAME, tie_weight=$tie_weight" >> "$LOG_FILE"
    else
        FAILED=$((FAILED + 1))
        echo "✗ 实验失败: $BACKBONE_NAME, tie_weight=$tie_weight (退出码: $EXIT_CODE)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [失败] 实验失败: $BACKBONE_NAME, tie_weight=$tie_weight (退出码: $EXIT_CODE)" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    echo ""
done

# ============================================
# Backbone 2: BAAI/bge-base-en-v1.5 (feature_fused model)
# ============================================
BACKBONE_NAME="bge-base-en-v1.5"
BACKBONE="BAAI/bge-base-en-v1.5"
MODEL_TYPE="feature_fused"
TRAINER_TYPE="feature_fused"
HIDDEN_SIZE="768"
LEARNING_RATE="7e-5"
TEMPERATURE="0.5"
EPOCHS="5"
WEIGHT_DECAY="0.01"

echo "----------------------------------------"
echo "Backbone 2: $BACKBONE_NAME (model_type=$MODEL_TYPE, lr=$LEARNING_RATE, wd=$WEIGHT_DECAY)"
echo "----------------------------------------"
echo ""

for tie_weight in "${TIE_WEIGHTS[@]}"; do
    # 生成输出目录名
    OUTPUT_DIR="$EXPERIMENTS_ROOT/${BACKBONE_NAME}/tie_weight_${tie_weight}"
    
    # 构建参数
    PARAMS=(
        "router/train_router.py"
        "--model_type" "$MODEL_TYPE"
        "--trainer_type" "$TRAINER_TYPE"
        "--backbone" "$BACKBONE"
        "--train_data" "$TRAIN_DATA"
        "--val_data" "$VAL_DATA"
        "--learning_rate" "$LEARNING_RATE"
        "--temperature" "$TEMPERATURE"
        "--epochs" "$EPOCHS"
        "--weight_decay" "$WEIGHT_DECAY"
        "--tie_weight" "$tie_weight"
        "--output_dir" "$OUTPUT_DIR"
    )
    
    # 显示当前实验信息
    PROGRESS="[$(($COMPLETED + $FAILED + 1))/$TOTAL] $BACKBONE_NAME, tie_weight=$tie_weight"
    echo "$PROGRESS"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始: $BACKBONE_NAME, tie_weight=$tie_weight" >> "$LOG_FILE"
    
    # 执行训练
    echo "命令: python ${PARAMS[@]}"
    python "${PARAMS[@]}"
    EXIT_CODE=$?
    
    # 检查结果
    if [ $EXIT_CODE -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "✓ 实验完成: $BACKBONE_NAME, tie_weight=$tie_weight"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [成功] 实验完成: $BACKBONE_NAME, tie_weight=$tie_weight" >> "$LOG_FILE"
    else
        FAILED=$((FAILED + 1))
        echo "✗ 实验失败: $BACKBONE_NAME, tie_weight=$tie_weight (退出码: $EXIT_CODE)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [失败] 实验失败: $BACKBONE_NAME, tie_weight=$tie_weight (退出码: $EXIT_CODE)" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    echo ""
done

# 统计信息
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "========================================"
echo "Tie权重搜索完成"
echo "========================================"
echo ""
echo "实验统计:"
echo "  总实验数: $TOTAL"
echo "  成功: $COMPLETED"
echo "  失败: $FAILED"
echo "  总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 搜索完成" >> "$LOG_FILE"
echo "总实验数: $TOTAL" >> "$LOG_FILE"
echo "成功: $COMPLETED" >> "$LOG_FILE"
echo "失败: $FAILED" >> "$LOG_FILE"
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "详细日志: $LOG_FILE" >> "$LOG_FILE"

echo "结果保存在: $EXPERIMENTS_ROOT"
echo "详细日志: $LOG_FILE"
echo ""
