#!/bin/bash
# ============================================
# End-to-End Router Test Script (Linux/WSL)
# ============================================
#
# Features:
# - Load trained Router model (DC/DPO/Internal Representation)
# - Route HotpotQA test data
# - Execute NoRAG and NaiveRAG strategies
# - Evaluate metrics (EM, F1, retrieval_time, generation_time)
#
# Usage:
#   chmod +x run_router_e2e_test.sh
#   ./run_router_e2e_test.sh
#   Or specify sample count:
#   ./run_router_e2e_test.sh --num_samples 100
#
# Internal Representation Router (预加载表征模式，推荐):
#   ./run_router_e2e_test.sh \
#       --router_type internal_representation \
#       --representations_path outputs/representations/test_1000 \
#       --questions_file HotpotQA/hotpot_dev_distractor_1000_samples.jsonl
#
# ============================================

set -e

# Default parameters
NUM_SAMPLES=1000
OUTPUT_FILE=""
MODEL_PATH=""
ROUTER_TYPE="auto"
REPRESENTATION_TYPE="deep_last_token"
LLM_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
LLM_DEVICE="auto"
LLM_DTYPE="float16"
# 预加载表征参数
REPRESENTATIONS_PATH="/home/lhz/code/routing_rag/outputs/representations/fp16_qwen2.5-3b-instruct_test1000"
QUESTIONS_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --router_type)
            ROUTER_TYPE="$2"
            shift 2
            ;;
        --representation_type)
            REPRESENTATION_TYPE="$2"
            shift 2
            ;;
        --llm_model_name)
            LLM_MODEL_NAME="$2"
            shift 2
            ;;
        --llm_device)
            LLM_DEVICE="$2"
            shift 2
            ;;
        --llm_dtype)
            LLM_DTYPE="$2"
            shift 2
            ;;
        --representations_path)
            REPRESENTATIONS_PATH="$2"
            shift 2
            ;;
        --questions_file)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Project root
PROJECT_ROOT="/home/lhz/code/routing_rag"
cd "$PROJECT_ROOT"

echo "========================================"
echo "End-to-End Router Test"
echo "========================================"
echo ""

# Generate output file name if not specified
if [ -z "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="evaluation_results/router_e2e_${TIMESTAMP}.json"
fi

# Default model path (internal representation router)
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="${PROJECT_ROOT}/outputs/internal_rep_router_experiments_5000/deep_last_token/final"
fi

# Default HotpotQA file
HOTPOTQA_FILE="${PROJECT_ROOT}/HotpotQA/hotpot_dev_distractor_1000_samples.jsonl"

# For preloaded representations mode, default questions_file to hotpotqa_file
if [ -n "$REPRESENTATIONS_PATH" ] && [ -z "$QUESTIONS_FILE" ]; then
    QUESTIONS_FILE="$HOTPOTQA_FILE"
fi

# Display configuration
echo "Configuration:"
echo "  Project Root: $PROJECT_ROOT"
echo "  Model Path: $MODEL_PATH"
echo "  Router Type: $ROUTER_TYPE"
echo "  Sample Count: $NUM_SAMPLES"
echo "  Output File: $OUTPUT_FILE"
echo ""

# For internal representation router
if [ "$ROUTER_TYPE" == "internal_representation" ] || [ "$ROUTER_TYPE" == "auto" ]; then
    echo "Internal Representation Router Settings:"
    echo "  Representation Type: $REPRESENTATION_TYPE"
    if [ -n "$REPRESENTATIONS_PATH" ]; then
        echo "  模式: 预加载表征"
        echo "  Representations Path: $REPRESENTATIONS_PATH"
        echo "  Questions File: $QUESTIONS_FILE"
    else
        echo "  模式: 实时提取表征"
        echo "  LLM Model: $LLM_MODEL_NAME"
        echo "  LLM Device: $LLM_DEVICE"
        echo "  LLM Dtype: $LLM_DTYPE"
    fi
    echo ""
fi

# Check required files
echo "Checking required files..."

NAIVE_RAG_INDEX="${PROJECT_ROOT}/naive_rag_index_hotpotqa_train_5000_samples"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Router model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$HOTPOTQA_FILE" ]; then
    echo "ERROR: HotpotQA data file not found: $HOTPOTQA_FILE"
    exit 1
fi

if [ ! -d "$NAIVE_RAG_INDEX" ]; then
    echo "ERROR: NaiveRAG index not found: $NAIVE_RAG_INDEX"
    exit 1
fi

echo "  [OK] Router Model: $MODEL_PATH"
echo "  [OK] HotpotQA Data: $HOTPOTQA_FILE"
echo "  [OK] NaiveRAG Index: $NAIVE_RAG_INDEX"
echo ""

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
fi

# Run test
echo "Starting test..."
echo "----------------------------------------"

START_TIME=$(date +%s)

# Build command
CMD="python tests/test_router_e2e.py \
    --model_path $MODEL_PATH \
    --hotpotqa_file $HOTPOTQA_FILE \
    --naive_rag_index_path $NAIVE_RAG_INDEX \
    --max_samples $NUM_SAMPLES \
    --output $OUTPUT_FILE \
    --router_type $ROUTER_TYPE \
    --representation_type $REPRESENTATION_TYPE"

# Add internal representation router parameters
if [ -n "$REPRESENTATIONS_PATH" ]; then
    CMD="$CMD --representations_path $REPRESENTATIONS_PATH"
fi
if [ -n "$QUESTIONS_FILE" ]; then
    CMD="$CMD --questions_file $QUESTIONS_FILE"
fi
if [ -n "$LLM_MODEL_NAME" ]; then
    CMD="$CMD --llm_model_name $LLM_MODEL_NAME"
fi
if [ -n "$LLM_DEVICE" ]; then
    CMD="$CMD --llm_device $LLM_DEVICE"
fi
if [ -n "$LLM_DTYPE" ]; then
    CMD="$CMD --llm_dtype $LLM_DTYPE"
fi

# Execute
$CMD

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$(echo "scale=2; $DURATION / 60" | bc)

echo ""
echo "========================================"
echo "Test Completed!"
echo "========================================"
echo "Total Duration: ${DURATION_MIN} minutes"
echo "Results File: $OUTPUT_FILE"
echo ""

# Display summary if output file exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Performance Summary:"
    
    python -c "
import json
import sys

try:
    with open('$OUTPUT_FILE', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    overall = metrics.get('overall', {})
    
    print(f\"  Samples: {overall.get('num_samples', 0)}\")
    print(f\"  EM: {overall.get('em', 0):.4f}\")
    print(f\"  F1: {overall.get('f1', 0):.4f}\")
    print(f\"  Avg Total Time: {overall.get('avg_total_time', 0):.4f}s\")
    print(f\"  Avg Retrieval Time: {overall.get('avg_retrieval_time', 0):.4f}s\")
    print(f\"  Avg Generation Time: {overall.get('avg_generation_time', 0):.4f}s\")
    print()
    
    by_strategy = metrics.get('by_strategy', {})
    print('Strategy Statistics:')
    for strategy, stats in by_strategy.items():
        print(f\"  {strategy}:\")
        print(f\"    Count: {stats.get('count', 0)} ({stats.get('ratio', 0):.2%})\")
        print(f\"    EM: {stats.get('em', 0):.4f}, F1: {stats.get('f1', 0):.4f}\")
        print(f\"    Avg Time: {stats.get('avg_total_time', 0):.4f}s\")
except Exception as e:
    print(f\"Error reading results: {e}\", file=sys.stderr)
    sys.exit(1)
"
fi

echo ""
echo "Tips:"
echo "  - Use --num_samples N to limit test samples"
echo "  - Use --router_type internal_representation for IR router"
echo "  - Use --representation_type to specify representation type"
echo ""
echo "  预加载表征模式（推荐）:"
echo "    ./scripts/test/run_router_e2e_test.sh \\"
echo "        --router_type internal_representation \\"
echo "        --num_samples 100 \\"
echo "        --representations_path outputs/representations/fp16_qwen2.5-3b-instruct_test1000 \\"
echo "        --questions_file HotpotQA/hotpot_dev_distractor_1000_samples.jsonl"
echo ""
echo "  实时提取模式（需要 GPU 加载 LLM）:"
echo "    ./scripts/test/run_router_e2e_test.sh \\"
echo "        --router_type internal_representation \\"
echo "        --num_samples 100 \\"
echo "        --llm_model_name Qwen/Qwen2.5-3B-Instruct"
