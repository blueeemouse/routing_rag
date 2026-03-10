#!/bin/bash
# DPO超参数快速搜索 - MiniLM-L6-v2 (2 epochs快速筛选)
#
# 用法:
#   bash scripts/search_param/search_dpo_minilm_beta_lr_quick.sh --gpu 3
#
# 说明:
#   - 仅训练2个epoch快速筛选最优beta和lr组合
#   - 找到最佳配置后，再用完整epoch训练

bash "$(dirname "$0")/search_dpo_minilm_beta_lr.sh" --epochs 2 "$@"
