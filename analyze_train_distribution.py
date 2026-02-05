"""
分析训练数据的标签分布
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router.trainable_router.config import TrainableRouterConfig
from router.trainable_router.datasets.hotpotqa_dataset import GenericRouterDataset

# 加载配置
config = TrainableRouterConfig.from_yaml("config/train_no_rag_vs_naive_improved.yaml")

# 创建数据集
dataset = GenericRouterDataset(config, tokenizer=None)
dataset.load_data(config.data.train_path)

print(f"\n{'='*80}")
print(f"训练数据标签分布统计")
print(f"{'='*80}")

# 统计标签分布
no_rag_count = 0
naive_rag_count = 0
total = len(dataset)

for i in range(len(dataset)):
    item = dataset[i]
    scores = item['scores']  # [no_rag_score, naive_rag_score]
    
    if scores[0] >= scores[1]:
        no_rag_count += 1
    else:
        naive_rag_count += 1

print(f"总样本数: {total}")
print(f"no_rag 标签: {no_rag_count} ({no_rag_count/total*100:.2f}%)")
print(f"naive_rag 标签: {naive_rag_count} ({naive_rag_count/total*100:.2f}%)")
print(f"{'='*80}\n")
