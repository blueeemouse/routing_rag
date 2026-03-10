#!/bin/bash
# DPO超参数细搜脚本 - MiniLM-L6-v2
# 在确定最佳LR数量级后，细化搜索beta和精确lr
#
# 用法:
#   bash scripts/search_param/search_dpo_minilm_fine_tune.sh --lr_base 1e-5 --gpu 3
#
# 说明:
#   - 在粗搜确定最佳lr数量级后（如 1e-5 最佳）
#   - 在该数量级附近细化搜索（如 5e-6, 1e-5, 2e-5）
#   - 同时更精细地搜索beta值

set -e

cd "$(dirname "$0")/../.."

# 默认参数
LR_BASE="1e-5"
EPOCHS=5
BATCH_SIZE=16
GPU_ID=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr_base)
            LR_BASE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
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
            echo "用法: $0 --lr_base 1e-5 [--epochs 5] [--gpu 3]"
            exit 1
            ;;
    esac
done

# 设置GPU
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "使用GPU: $GPU_ID"
fi

# 基于lr_base生成细搜范围
# 如 1e-5 -> 5e-6, 1e-5, 2e-5
if [[ "$LR_BASE" == "1e-5" ]]; then
    LEARNING_RATES=(5e-6 1e-5 2e-5)
elif [[ "$LR_BASE" == "1e-4" ]]; then
    LEARNING_RATES=(5e-5 1e-4 2e-4)
elif [[ "$LR_BASE" == "1e-6" ]]; then
    LEARNING_RATES=(5e-7 1e-6 2e-6)
else
    # 通用计算：lr_base/2, lr_base, lr_base*2
    python3 << EOF
lr = float('$LR_BASE')
print(f"{lr/2:.0e}")
print(f"{lr:.0e}")
print(f"{lr*2:.0e}")
EOF
    LEARNING_RATES=($(python3 -c "lr=float('$LR_BASE'); print(f'{lr/2:.0e} {lr:.0e} {lr*2:.0e}')"))
fi

# Beta细搜（在粗搜最佳值附近）
BETAS=(0.01 0.03 0.05 0.07 0.1)

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
# 使用小数据集(5000条)加速超参数搜索
TRAIN_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_full.json"
VAL_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_val.json"
BASE_OUTPUT_DIR="router_models/dpo_search_minilm_fine"

mkdir -p "$BASE_OUTPUT_DIR"

echo "=========================================="
echo "DPO Fine-Grained Search"
echo "LR Base: $LR_BASE -> [${LEARNING_RATES[*]}]"
echo "Betas: [${BETAS[*]}]"
echo "=========================================="

RESULTS_FILE="$BASE_OUTPUT_DIR/search_results_fine.txt"
echo "Fine Search (LR base: $LR_BASE)" > "$RESULTS_FILE"
printf "%-10s %-15s %-20s %-15s\n" "Beta" "Learning Rate" "Output Dir" "Best Val Acc" >> "$RESULTS_FILE"
echo "----------------------------------------" >> "$RESULTS_FILE"

TOTAL=$(( ${#BETAS[@]} * ${#LEARNING_RATES[@]} ))
CURRENT=0

for BETA in "${BETAS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        EXP_NAME="fine_beta${BETA}_lr${LR}"
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXP_NAME"
        
        echo ""
        echo "[$CURRENT/$TOTAL] Beta=$BETA, LR=$LR"
        
        if [[ -f "$OUTPUT_DIR/eval_result.json" ]]; then
            echo "已存在，跳过..."
        else
            python router/train_dpo.py \
                --model_name "$MODEL_NAME" \
                --train_file "$TRAIN_FILE" \
                --val_file "$VAL_FILE" \
                --output_dir "$OUTPUT_DIR" \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --learning_rate $LR \
                --beta $BETA \
                --max_length 512 \
                --logging_steps 50 \
                --eval_steps 100 \
                --save_total_limit 1 \
                --warmup_steps 100 \
                --fp16 \
                2>&1 | tee "$OUTPUT_DIR/training_console.log"
        fi
        
        if [[ -f "$OUTPUT_DIR/best_model_info.json" ]]; then
            BEST_ACC=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/best_model_info.json')).get('eval_accuracy', 'N/A'))")
        else
            BEST_ACC="N/A"
        fi
        
        printf "%-10s %-15s %-20s %-15s\n" "$BETA" "$LR" "$EXP_NAME" "$BEST_ACC" >> "$RESULTS_FILE"
        echo "  Result: $BEST_ACC"
    done
done

echo ""
cat "$RESULTS_FILE"

# 汇总最佳
python3 << 'EOF'
import json
from pathlib import Path

base_dir = Path("router_models/dpo_search_minilm_fine")
results = []

for exp_dir in base_dir.iterdir():
    if not exp_dir.is_dir() or not exp_dir.name.startswith('fine_'):
        continue
    
    best_acc = None
    for fname in ['best_model_info.json', 'eval_result.json']:
        fpath = exp_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
                best_acc = data.get('eval_accuracy') or data.get('accuracy')
                if best_acc:
                    break
    
    if best_acc:
        name = exp_dir.name.replace('fine_beta', '').replace('_lr', ' ')
        parts = name.split()
        if len(parts) >= 2:
            results.append({'beta': parts[0], 'lr': parts[1], 'acc': float(best_acc)})

if results:
    best = max(results, key=lambda x: x['acc'])
    print(f"\n最佳细搜配置: Beta={best['beta']}, LR={best['lr']}, Acc={best['acc']:.4f}")
EOF
