#!/bin/bash
# 批量训练 Hybrid Representation Fusion Router
# 骨干: BAAI/bge-base-en-v1.5 (768-d) + 不同内部表征类型 + Cross Attention 融合
#
# 使用方法:
#   bash scripts/train/batch_train_hybrid_fusion_bge.sh              # 训练所有组合
#   bash scripts/train/batch_train_hybrid_fusion_bge.sh --quick-test  # 快速测试 (10 steps each)

set -e

# ===================== 固定参数 =====================
CONFIG="config/train_hybrid_representation_fusion.yaml"
BACKBONE="BAAI/bge-base-en-v1.5"
LEARNING_RATE=7e-5
BATCH_SIZE=32
WEIGHT_DECAY=0.01
EPOCHS=10
EVAL_STEPS=50
SEED=42
FUSION_TYPE="cross_attn"

# 数据路径
TRAIN_REPR_DIR="outputs/representations/fp16_qwen2.5-3b-instruct_train5000"
VAL_REPR_DIR="outputs/representations/fp16_qwen2.5-3b-instruct_test1000"
LABELS_PATH="HotpotQA_train_data/label_analysis/all_labels_vllm_qwen_with_tie_converted.json"

# 根输出目录 (按骨架和时间戳区分)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUTPUT_DIR="outputs/hybrid_fusion_bge-base/${TIMESTAMP}"

# ===================== 内部表征类型列表 =====================
# 格式: "类型:维度"
REPRESENTATIONS=(
    "shallow_mean:2048"
    "shallow_last_token:2048"
    "middle_mean:2048"
    "middle_last_token:2048"
    "deep_mean:2048"
    "deep_last_token:2048"
    "concat_deep:4096"
    "concat_all_mean:6144"
    "concat_all_last:6144"
    "concat_all:12288"
)

# ===================== 快速测试标志 =====================
QUICK_TEST=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "批量训练 hybrid fusion (bge-base + 内部表征) 路由器"
            echo ""
            echo "Options:"
            echo "  --quick-test    快速测试模式 (每个配置只跑 10 steps)"
            echo "  -h, --help      显示帮助"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ===================== 颜色输出 =====================
echo_cyan()  { echo -e "\033[36m$1\033[0m"; }
echo_green() { echo -e "\033[32m$1\033[0m"; }
echo_yellow(){ echo -e "\033[33m$1\033[0m"; }
echo_red()   { echo -e "\033[31m$1\033[0m"; }

# ===================== 打印总览 =====================
echo_cyan "========================================="
echo_cyan "批量训练 Hybrid Fusion (bge-base + 内部表征)"
echo_cyan "========================================="
echo "骨干模型:    $BACKBONE"
echo "学习率:      $LEARNING_RATE"
echo "Batch Size:  $BATCH_SIZE"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Epochs:      $EPOCHS"
echo "融合方式:    $FUSION_TYPE"
echo ""
echo "训练数据:    $TRAIN_REPR_DIR"
echo "验证数据:    $VAL_REPR_DIR"
echo "标签文件:    $LABELS_PATH"
echo ""
echo "根输出目录:  $ROOT_OUTPUT_DIR"
echo ""
echo "表征组合 (共 ${#REPRESENTATIONS[@]} 个):"
for item in "${REPRESENTATIONS[@]}"; do
    echo "  - $item"
done
echo_cyan "========================================="

# ===================== 训练结果汇总 =====================
RESULTS_FILE="${ROOT_OUTPUT_DIR}/_summary.txt"
mkdir -p "$ROOT_OUTPUT_DIR"
echo "# Hybrid Fusion (bge-base) 批量训练结果汇总" > "$RESULTS_FILE"
echo "# 时间: $(date)" >> "$RESULTS_FILE"
echo "# 骨干: $BACKBONE" >> "$RESULTS_FILE"
echo "# 学习率: $LEARNING_RATE, Batch Size: $BATCH_SIZE, Weight Decay: $WEIGHT_DECAY" >> "$RESULTS_FILE"
echo "# Fusion: $FUSION_TYPE" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
printf "%-30s %-10s %-12s %s\n" "表征类型" "维度" "输出目录" "状态" >> "$RESULTS_FILE"
echo "------------------------------------------------------------" >> "$RESULTS_FILE"

# ===================== 逐个训练 =====================
TOTAL=${#REPRESENTATIONS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!REPRESENTATIONS[@]}"; do
    ITEM="${REPRESENTATIONS[$i]}"
    REPR_TYPE="${ITEM%%:*}"
    DIM="${ITEM##*:}"
    IDX=$((i + 1))

    OUTPUT_DIR="${ROOT_OUTPUT_DIR}/${REPR_TYPE}"
    LOG_FILE="${ROOT_OUTPUT_DIR}/${REPR_TYPE}.log"

    echo ""
    echo_cyan "---------------------------------------------------------"
    echo_cyan "[${IDX}/${TOTAL}] 训练: ${REPR_TYPE} (dim=${DIM})"
    echo_cyan "---------------------------------------------------------"
    echo "  输出目录: $OUTPUT_DIR"

    # 构建命令
    COMMAND="python router/train_router.py"
    COMMAND="$COMMAND --config $CONFIG"
    COMMAND="$COMMAND --output_dir $OUTPUT_DIR"
    COMMAND="$COMMAND --backbone $BACKBONE"
    COMMAND="$COMMAND --representation_type $REPR_TYPE"
    COMMAND="$COMMAND --representation_dim $DIM"
    COMMAND="$COMMAND --representation_dir $TRAIN_REPR_DIR"
    COMMAND="$COMMAND --labels_path $LABELS_PATH"
    # val_representation_dir 使用 YAML 配置中的默认值
    COMMAND="$COMMAND --fusion_type $FUSION_TYPE"
    COMMAND="$COMMAND --learning_rate $LEARNING_RATE"
    COMMAND="$COMMAND --batch_size $BATCH_SIZE"
    COMMAND="$COMMAND --weight_decay $WEIGHT_DECAY"
    COMMAND="$COMMAND --epochs $EPOCHS"
    COMMAND="$COMMAND --eval_steps $EVAL_STEPS"
    COMMAND="$COMMAND --seed $SEED"

    if [[ "$QUICK_TEST" == true ]]; then
        COMMAND="$COMMAND --max_steps 10 --overfit_single_batch"
    fi

    # 执行训练
    STATUS="OK"
    if eval $COMMAND 2>&1 | tee "$LOG_FILE"; then
        echo_green "[${IDX}/${TOTAL}] ${REPR_TYPE} 训练完成"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        STATUS="FAILED"
        echo_red "[${IDX}/${TOTAL}] ${REPR_TYPE} 训练失败 (详见 ${LOG_FILE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # 记录汇总
    printf "%-30s %-10s %-12s %s\n" "$REPR_TYPE" "$DIM" "$REPR_TYPE" "$STATUS" >> "$RESULTS_FILE"
done

# ===================== 最终汇总 =====================
echo ""
echo_cyan "========================================="
echo_cyan "批量训练完成"
echo_cyan "========================================="
echo "总任务数:   ${TOTAL}"
echo_green "成功:       ${SUCCESS_COUNT}"
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo_red "失败:       ${FAIL_COUNT}"
fi
echo ""
echo "根输出目录: $ROOT_OUTPUT_DIR"
echo "结果汇总:   $RESULTS_FILE"
echo ""
echo "各实验目录结构:"
echo "  ${ROOT_OUTPUT_DIR}/"
echo "  ├── shallow_mean/"
echo "  ├── shallow_last_token/"
echo "  ├── ..."
echo "  ├── concat_all/"
echo "  ├── _summary.txt"
echo "  ├── shallow_mean.log"
echo "  └── ..."
echo_cyan "========================================="
