"""
诊断训练好的router模型

功能：
1. 检查策略embedding的分布
2. 在数据集上评估
3. 分析不同分数范围的预测分布
"""

import torch
import json
import os
import sys
import numpy as np
from typing import Dict, List, Any

# 添加项目根目录到路径
ROUTING_RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROUTING_RAG_ROOT)

from router.trainable_router.factory import TrainableRouterFactory
from router.trainable_router.config import TrainableRouterConfig
from router.trainable_router.data_utils import DataAdapter


def load_model(model_path: str):
    """
    加载训练好的模型

    Args:
        model_path: 模型路径

    Returns:
        模型和配置
    """
    # 加载配置
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    # 创建模型
    config = TrainableRouterConfig()
    config.model.strategy_names = model_config['strategy_names']
    config.model.hidden_size = model_config['hidden_size']
    config.model.num_strategies = model_config['num_strategies']
    config.model.similarity_function = "cos"
    config.model.temperature = 1.0  # 训练时的温度
    config.model.backbone_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.model.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model.device = config.device

    model = TrainableRouterFactory.create_model(config, model_path=model_path)

    return model, config


def analyze_strategy_embeddings(model):
    """
    分析策略embedding的分布

    Args:
        model: 路由器模型
    """
    print("="*80)
    print("策略Embedding分析")
    print("="*80)

    strategy_embs = model.get_strategy_embeddings()
    strategy_names = model.strategy_names

    print(f"\n策略名称: {strategy_names}")
    print(f"Embedding维度: {strategy_embs.shape}")

    print("\n每个策略的Embedding统计:")
    for i, name in enumerate(strategy_names):
        emb = strategy_embs[i]
        print(f"\n{name}:")
        print(f"  均值: {emb.mean().item():.6f}")
        print(f"  标准差: {emb.std().item():.6f}")
        print(f"  最小值: {emb.min().item():.6f}")
        print(f"  最大值: {emb.max().item():.6f}")
        print(f"  L2范数: {emb.norm().item():.6f}")

    # 计算策略之间的相似度
    print("\n策略之间的余弦相似度:")
    for i in range(len(strategy_names)):
        for j in range(i+1, len(strategy_names)):
            sim = torch.nn.functional.cosine_similarity(
                strategy_embs[i:i+1],
                strategy_embs[j:j+1]
            ).item()
            print(f"  {strategy_names[i]} vs {strategy_names[j]}: {sim:.6f}")


def load_and_evaluate(model, data_path: str, num_samples: int = 1000):
    """
    加载数据并评估模型

    Args:
        model: 路由器模型
        data_path: 数据路径
        num_samples: 评估样本数量

    Returns:
        评估结果
    """
    print("\n" + "="*80)
    print("模型评估")
    print("="*80)

    # 加载数据
    adapter = DataAdapter("em * 0.5 + f1 * 0.5")

    if os.path.isdir(data_path):
        items = adapter.from_directory(data_path)
        items = adapter.aggregate(items)
        items = adapter.filter_by_coverage(items, min_strategies=2)
    else:
        raise ValueError(f"无效的数据路径: {data_path}")

    print(f"\n加载了 {len(items)} 条数据")

    # 评估
    model.eval()

    queries = []
    true_labels = []
    pred_labels = []
    no_rag_scores = []
    naive_rag_scores = []

    with torch.no_grad():
        for i, item in enumerate(items[:num_samples]):
            if i % 100 == 0:
                print(f"已处理 {i}/{min(num_samples, len(items))}...")

            # 获取问题和真实分数
            query = item.question
            queries.append(query)

            # 计算各策略的分数
            score_computer = adapter.score_computer
            no_rag_score = score_computer.compute(item.strategy_scores.get('no_rag', {}))
            naive_rag_score = score_computer.compute(item.strategy_scores.get('naive_rag', {}))
            no_rag_scores.append(no_rag_score)
            naive_rag_scores.append(naive_rag_score)

            # 确定真实标签（最高分）
            if naive_rag_score > no_rag_score:
                true_label = 'naive_rag'
            elif no_rag_score > naive_rag_score:
                true_label = 'no_rag'
            else:
                # 分数相等，按顺序
                true_label = 'no_rag'
            true_labels.append(true_label)

            # 预测
            pred = model.route([query])[0]
            pred_labels.append(pred)

    # 计算准确率
    correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    accuracy = correct / len(true_labels)

    # 统计预测分布
    pred_dist = {}
    for label in pred_labels:
        pred_dist[label] = pred_dist.get(label, 0) + 1

    true_dist = {}
    for label in true_labels:
        true_dist[label] = true_dist.get(label, 0) + 1

    print(f"\n评估结果:")
    print(f"  样本数: {len(pred_labels)}")
    print(f"  准确率: {accuracy:.4f} ({correct}/{len(pred_labels)})")
    print(f"\n真实标签分布:")
    for label, count in true_dist.items():
        print(f"  {label}: {count} ({count/len(true_labels)*100:.2f}%)")
    print(f"\n预测标签分布:")
    for label, count in pred_dist.items():
        print(f"  {label}: {count} ({count/len(pred_labels)*100:.2f}%)")

    # 按分数范围分析
    print("\n" + "="*80)
    print("按分数范围分析")
    print("="*80)

    # 按分数差分组
    score_diffs = [n - s for n, s in zip(naive_rag_scores, no_rag_scores)]
    ranges = [
        (-1.0, -0.5, "NaiveRAG明显优势"),
        (-0.5, 0.0, "NaiveRAG轻微优势"),
        (0.0, 0.0, "分数相等"),
        (0.0, 0.5, "NoRAG轻微优势"),
        (0.5, 1.0, "NoRAG明显优势"),
    ]

    for low, high, desc in ranges:
        mask = [low < diff <= high for diff in score_diffs]
        if not any(mask):
            continue

        indices = [i for i, m in enumerate(mask) if m]
        if not indices:
            continue

        range_correct = sum(1 for i in indices if pred_labels[i] == true_labels[i])
        range_accuracy = range_correct / len(indices) if indices else 0

        print(f"\n{desc} (分数差在({low}, {high}]):")
        print(f"  样本数: {len(indices)}")
        print(f"  准确率: {range_accuracy:.4f}")

        # 显示前几个例子
        print(f"  示例:")
        for idx in indices[:3]:
            print(f"    {queries[idx][:60]}...")
            print(f"      真实: {true_labels[idx]}, 预测: {pred_labels[idx]}")
            print(f"      分数: NoRAG={no_rag_scores[idx]:.3f}, NaiveRAG={naive_rag_scores[idx]:.3f}")

    return {
        'accuracy': accuracy,
        'pred_dist': pred_dist,
        'true_dist': true_dist,
        'num_samples': len(pred_labels)
    }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='诊断训练好的router模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--data_path', type=str,
                       default='HotpotQA_train_data',
                       help='评估数据路径')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='评估样本数量')
    parser.add_argument('--output_dir', type=str,
                       default='router_diagnosis',
                       help='诊断结果输出目录')

    args = parser.parse_args()

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model, config = load_model(args.model_path)

    # 分析策略embedding
    analyze_strategy_embeddings(model)

    # 评估模型
    results = load_and_evaluate(model, args.data_path, args.num_samples)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, 'diagnosis_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n诊断结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
