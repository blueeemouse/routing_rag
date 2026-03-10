#!/bin/bash
# DPO超参数搜索脚本(第1阶段-粗搜) - MiniLM-L6-v2
# 搜索参数: beta (DPO温度) 和 learning_rate (数量级)
#
# 用法:
#   bash scripts/search_param/search_dpo_minilm_beta_lr.sh                    # 使用所有GPU
#   bash scripts/search_param/search_dpo_minilm_beta_lr.sh --gpu 3            # 只使用3号GPU
#   bash scripts/search_param/search_dpo_minilm_beta_lr.sh --epochs 2         # 快速筛选
#
# 搜索策略(两阶段):
#   第1阶段(本脚本): 粗搜确定LR数量级
#     - Learning Rate: 1e-6, 1e-5, 1e-4 (跨数量级)
#     - Beta: 0.01, 0.05, 0.1, 0.2 (常见范围)
#     - 共12组实验，建议用 --epochs 2 快速筛选
#
#   第2阶段: 细搜最佳LR附近 + Beta精确值
#     - 使用 search_dpo_minilm_fine_tune.sh
#     - 如最佳LR=1e-5，则细搜: 5e-6, 1e-5, 2e-5
#     - Beta细搜: 0.01, 0.03, 0.05, 0.07, 0.1
#
# 说明:
#   - beta: 控制策略偏离参考模型的程度，分类任务建议 0.01~0.2
#   - learning_rate: 先定数量级再细化，MiniLM常用 1e-5

set -e  # 遇到错误立即退出，便于调试

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "DPO Hyperparameter Search - MiniLM-L6-v2"
echo "=========================================="
echo ""

# 可配置参数
EPOCHS=5
BATCH_SIZE=16
GPU_ID="0,1,2"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
            exit 1
            ;;
    esac
done

# 设置GPU
if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "使用GPU: $GPU_ID"
fi

# 搜索空间 - 先粗粒度搜索
# Beta: DPO温度系数，控制策略偏离参考模型的程度
#   - 常见范围: 0.01 ~ 0.5
#   - 分类任务通常用较小值: 0.01 ~ 0.2
#   - 不需要跨数量级搜索，在常见值周围密集采样
BETAS=(0.01 0.05 0.1 0.2)

# Learning Rate: 先确定数量级，再细化
#   - 1e-4: 较大的学习率，可能不稳定
#   - 1e-5: 常用起始点（MiniLM推荐）
#   - 1e-6: 保守设置，需要更多epoch
LEARNING_RATES=(1e-4 1e-5 1e-6)

# 固定参数
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
# 使用小数据集(5000条)加速超参数搜索
TRAIN_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_full.json"
VAL_FILE="HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_val.json"
BASE_OUTPUT_DIR="router_models/dpo_search_minilm_beta_lr"

# 创建基础目录
mkdir -p "$BASE_OUTPUT_DIR"

# 记录搜索结果
RESULTS_FILE="$BASE_OUTPUT_DIR/search_results.txt"
echo "DPO Hyperparameter Search Results" > "$RESULTS_FILE"
echo "==================================" >> "$RESULTS_FILE"
echo "Model: $MODEL_NAME" >> "$RESULTS_FILE"
echo "Epochs: $EPOCHS" >> "$RESULTS_FILE"
echo "Batch Size: $BATCH_SIZE" >> "$RESULTS_FILE"
echo "Search Space:" >> "$RESULTS_FILE"
echo "  Beta: ${BETAS[*]}" >> "$RESULTS_FILE"
echo "  Learning Rate: ${LEARNING_RATES[*]}" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Results:" >> "$RESULTS_FILE"
printf "%-10s %-15s %-20s %-15s\n" "Beta" "Learning Rate" "Output Dir" "Best Val Acc" >> "$RESULTS_FILE"
echo "----------------------------------------" >> "$RESULTS_FILE"

# 总实验数
TOTAL=$(( ${#BETAS[@]} * ${#LEARNING_RATES[@]} ))
CURRENT=0

# 网格搜索
for BETA in "${BETAS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # 创建实验目录名
        EXP_NAME="beta${BETA}_lr${LR}"
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXP_NAME"
        
        echo ""
        echo "=========================================="
        echo "Experiment $CURRENT/$TOTAL: Beta=$BETA, LR=$LR"
        echo "Output: $OUTPUT_DIR"
        echo "=========================================="
        
        # 检查是否已存在结果
        if [[ -f "$OUTPUT_DIR/eval_result.json" ]]; then
            echo "实验已存在，跳过训练，直接读取结果..."
        else
            # 运行训练 (不使用管道，避免set -e问题)
            echo "开始训练实验 $CURRENT/$TOTAL ..."
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
                --save_steps 1000 \
                --save_total_limit 1 \
                --warmup_steps 100 \
                --fp16 2>&1 | tee "$OUTPUT_DIR/training_console.log"
            TRAIN_EXIT_CODE=${PIPESTATUS[0]}
            if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
                echo "警告: 训练实验 $CURRENT 退出码为 $TRAIN_EXIT_CODE"
            fi
        fi
        
        # 读取最佳验证准确率
        BEST_ACC="N/A"
        if [[ -f "$OUTPUT_DIR/best_model_info.json" ]]; then
            BEST_ACC=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/best_model_info.json')); print(data.get('eval_accuracy', 'N/A'))" 2>/dev/null || echo "N/A")
        elif [[ -f "$OUTPUT_DIR/eval_result.json" ]]; then
            BEST_ACC=$(python3 -c "import json; data=json.load(open('$OUTPUT_DIR/eval_result.json')); print(data.get('accuracy', 'N/A'))" 2>/dev/null || echo "N/A")
        fi
        
        # 记录结果
        printf "%-10s %-15s %-20s %-15s\n" "$BETA" "$LR" "$EXP_NAME" "$BEST_ACC" >> "$RESULTS_FILE"
        echo "  Best Val Acc: $BEST_ACC"
    done
done

# 汇总最佳配置
echo ""
echo "=========================================="
echo "搜索完成！汇总结果:"
echo "=========================================="
cat "$RESULTS_FILE"

# 找出最佳配置
python3 << 'PYEOF' || echo "警告: 汇总结果时出错"
import json
import os
from pathlib import Path

try:
    base_dir = Path("router_models/dpo_search_minilm_beta_lr")
    results = []
    
    if not base_dir.exists():
        print(f"目录不存在: {base_dir}")
        exit(0)
    
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        best_acc = None
        
        # 尝试读取best_model_info.json
        best_info_file = exp_dir / "best_model_info.json"
        if best_info_file.exists():
            try:
                with open(best_info_file) as f:
                    data = json.load(f)
                    best_acc = data.get('eval_accuracy')
            except:
                pass
        
        # 回退到eval_result.json
        if best_acc is None:
            eval_file = exp_dir / "eval_result.json"
            if eval_file.exists():
                try:
                    with open(eval_file) as f:
                        data = json.load(f)
                        best_acc = data.get('accuracy')
                except:
                    pass
        
        if best_acc is not None:
            # 从目录名解析参数
            exp_name = exp_dir.name
            try:
                parts = exp_name.replace('beta', '').split('_lr')
                if len(parts) == 2:
                    beta, lr = parts[0], parts[1]
                    results.append({
                        'beta': float(beta),
                        'lr': float(lr),
                        'accuracy': float(best_acc),
                        'dir': str(exp_dir)
                    })
            except:
                pass
    
    if results:
        best = max(results, key=lambda x: x['accuracy'])
        print(f"\n最佳配置:")
        print(f"  Beta: {best['beta']}")
        print(f"  Learning Rate: {best['lr']}")
        print(f"  Validation Accuracy: {best['accuracy']:.4f}")
        print(f"  Model Path: {best['dir']}")
        
        # 保存最佳配置
        with open(base_dir / "best_config.json", 'w') as f:
            json.dump(best, f, indent=2)
        print(f"\n最佳配置已保存到: {base_dir / 'best_config.json'}")
    else:
        print("未找到有效的实验结果")
except Exception as e:
    print(f"汇总结果时出错: {e}")
PYEOF

echo ""
echo "所有实验结果保存在: $BASE_OUTPUT_DIR"
echo "详细日志: $RESULTS_FILE"
