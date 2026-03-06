#!/bin/bash
# GraphRAG Standard 模式索引构建脚本
# 使用 Standard 模式构建 GraphRAG 索引

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 项目根目录
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 配置参数
WORK_DIR="/home/lhz/code/routing_rag/graphrag_index_hotpotqa_train_5000_samples"
CONFIG_FILE="graphrag_hotpotqa_config.yml"
HOTPOTQA_FILE="/home/lhz/code/routing_rag/HotpotQA/hotpot_train_v1.1_5000_samples.jsonl"
NUM_SAMPLES=5000

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 调用 Python 脚本
python ./scripts/build_index/build_graphrag_index.py \
    --work_dir "$WORK_DIR" \
    --config_file "$CONFIG_FILE" \
    --method standard \
    --hotpotqa_file "$HOTPOTQA_FILE" \
    --num_samples $NUM_SAMPLES
