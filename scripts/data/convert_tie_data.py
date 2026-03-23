#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建完整5000条训练数据（含tie样本转换）

目的：
    将 all_labels.json 转换为 RouterLabelDataset 兼容格式
    
数据转换规则：
    - tie → no_rag（效率优先：性能相同时选择更快的方法）
    - naive_rag → naive_rag
    - no_rag → no_rag

输入：
    all_labels.json: 原始5000条数据，包含 tie 标签
    
输出：
    all_labels_with_tie_converted.json: 转换后的数据，兼容 RouterLabelDataset

数据分布：
    原始：
      - tie: 2549
      - naive_rag: 2090
      - no_rag: 361
    
    转换后：
      - no_rag: 2910 (361 + 2549)
      - naive_rag: 2090
      - 总计: 5000
"""

import json
import os
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description='转换 tie 样本为 no_rag')
    parser.add_argument('--input', type=str,
                        default="/home/lhz/code/routing_rag/HotpotQA_train_data/label_analysis/all_labels_vllm_qwen.json",
                        help='输入的 all_labels.json 文件路径')
    parser.add_argument('--output', type=str,
                        default="/home/lhz/code/routing_rag/HotpotQA_train_data/label_analysis/all_labels_vllm_qwen_with_tie_converted.json",
                        help='输出文件路径')
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # 加载原始数据
    print(f"加载原始数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_data = data.get('samples', [])
    print(f"原始数据条数: {len(raw_data)}")
    
    # 统计原始标签分布
    raw_labels = Counter(s['optimal_strategy'] for s in raw_data)
    print(f"原始标签分布: {raw_labels}")
    
    # 转换数据
    samples = []
    tie_count = 0
    
    for item in raw_data:
        label = item['optimal_strategy']
        
        # tie → no_rag（效率优先）
        if label == 'tie':
            optimal_strategy = 'no_rag'
            tie_count += 1
        else:
            optimal_strategy = label
        
        sample = {
            'question': item['question'],
            'optimal_strategy': optimal_strategy,
            'no_rag_score': item.get('no_rag_score', 0.0),
            'naive_rag_score': item.get('naive_rag_score', 0.0),
            'source': 'tie_converted' if label == 'tie' else 'original'
        }
        samples.append(sample)
    
    # 统计转换后分布
    new_labels = Counter(s['optimal_strategy'] for s in samples)
    print(f"\n转换后标签分布: {new_labels}")
    print(f"  - tie 样本转换为 no_rag: {tie_count} 条")
    
    # 保存
    output_data = {'samples': samples}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到: {output_path}")
    
    # 验证
    with open(output_path, 'r', encoding='utf-8') as f:
        verify = json.load(f)
    print(f"验证: {len(verify['samples'])} 条样本")

if __name__ == '__main__':
    main()
