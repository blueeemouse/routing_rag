"""
计算软标签 (含成本考量)

软标签公式:
- Q = alpha * F1 + (1-alpha) * EM  (质量分数, 默认 alpha=0.8)
- Cost_norm = Cost / max_cost  (归一化成本)
- U = Q - lambda * Cost_norm  (效用函数)
- ΔU = U_naive_rag - U_no_rag
- soft_label = sigmoid(ΔU / τ) = 1 / (1 + exp(-ΔU / τ))

其中:
- no_rag 的 Cost = 0 (无检索)
- naive_rag 的 Cost = retrieval_time

soft_label 含义:
- 接近 0: no_rag 更好 (考虑成本后)
- 接近 0.5: tie (两者差不多)
- 接近 1: naive_rag 更好 (即使考虑成本也值得)

注意: 引入成本后, tie 样本会倾向于 no_rag (因为同性能下 no_rag 成本更低)
"""

import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_results_with_time(results_file: str) -> Dict[str, Dict[str, float]]:
    """
    从结果文件加载 retrieval_time 信息
    
    Args:
        results_file: 结果文件路径 (Norag_results_*.json 或 Naiverag_results_*.json)
    
    Returns:
        {question: {"retrieval_time": float, "total_time": float}}
    """
    print(f"读取结果文件: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    predictions = data.get('results', {}).get('predictions', [])
    
    for pred in predictions:
        question = pred.get('question', '')
        if question:
            results[question] = {
                'retrieval_time': pred.get('retrieval_time', 0.0),
                'total_time': pred.get('total_time', 0.0),
            }
    
    print(f"  加载了 {len(results)} 条记录")
    return results


def compute_soft_label_with_cost(
    no_rag_f1: float,
    no_rag_em: float,
    naive_rag_f1: float,
    naive_rag_em: float,
    naive_rag_retrieval_time: float,
    max_retrieval_time: float,
    alpha: float = 0.8,
    lambda_cost: float = 0.5,
    temperature: float = 0.1,
) -> tuple:
    """
    计算软标签 (含成本考量)
    
    Args:
        no_rag_f1, no_rag_em: no_rag 的 F1 和 EM 分数
        naive_rag_f1, naive_rag_em: naive_rag 的 F1 和 EM 分数
        naive_rag_retrieval_time: naive_rag 的检索时间
        max_retrieval_time: 用于归一化的最大检索时间
        alpha: F1 权重 (默认 0.8, 即 Q = 0.8*F1 + 0.2*EM)
        lambda_cost: 成本权重 (默认 0.5)
        temperature: 温度参数 (越小越接近硬标签)
    
    Returns:
        (soft_label, utility_gap, Q_no_rag, Q_naive_rag, U_no_rag, U_naive_rag, cost_norm)
    """
    # 质量分数
    Q_no_rag = alpha * no_rag_f1 + (1 - alpha) * no_rag_em
    Q_naive_rag = alpha * naive_rag_f1 + (1 - alpha) * naive_rag_em
    
    # 归一化成本
    cost_norm = 0.0
    if max_retrieval_time > 0:
        cost_norm = naive_rag_retrieval_time / max_retrieval_time
    
    # Utility = Q - lambda * Cost_norm
    # no_rag 的成本为 0
    U_no_rag = Q_no_rag
    U_naive_rag = Q_naive_rag - lambda_cost * cost_norm
    
    # Utility gap (naive_rag 相对于 no_rag 的优势)
    utility_gap = U_naive_rag - U_no_rag
    
    # 软标签 (sigmoid)
    if temperature <= 0:
        temperature = 0.001  # 防止除零
    
    soft_label = 1.0 / (1.0 + math.exp(-utility_gap / temperature))
    
    return soft_label, utility_gap, Q_no_rag, Q_naive_rag, U_no_rag, U_naive_rag, cost_norm


def process_labels_with_cost(
    labels_file: str,
    naive_rag_results_file: str,
    output_file: str,
    alpha: float = 0.8,
    lambda_cost: float = 0.5,
    temperature: float = 0.1,
):
    """
    处理标签文件, 添加软标签 (含成本考量)
    
    Args:
        labels_file: 输入的 all_labels.json 文件
        naive_rag_results_file: naive_rag 结果文件 (包含 retrieval_time)
        output_file: 输出文件
        alpha: F1 权重
        lambda_cost: 成本权重
        temperature: 温度参数
    """
    # 加载 naive_rag 结果 (获取检索时间)
    naive_rag_results = load_results_with_time(naive_rag_results_file)
    
    # 加载标签数据
    print(f"读取标签文件: {labels_file}")
    with open(labels_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    else:
        samples = data
    
    print(f"共 {len(samples)} 条样本")
    
    # 第一遍: 找到最大检索时间 (用于归一化)
    max_retrieval_time = 0.0
    for sample in samples:
        question = sample.get('question', '')
        if question in naive_rag_results:
            rt = naive_rag_results[question].get('retrieval_time', 0.0)
            if rt > max_retrieval_time:
                max_retrieval_time = rt
    
    print(f"最大检索时间: {max_retrieval_time:.4f} 秒")
    
    # 第二遍: 计算软标签
    processed_samples = []
    stats = {
        'total': 0,
        'soft_label_near_0': 0,  # < 0.3
        'soft_label_near_05': 0,  # 0.3 ~ 0.7
        'soft_label_near_1': 0,  # > 0.7
        'missing_time': 0,  # 缺少时间信息的样本
    }
    
    # 额外统计
    sum_utility_gap = 0.0
    sum_cost_norm = 0.0
    
    for sample in samples:
        question = sample.get('question', '')
        
        # 获取分数
        no_rag_f1 = sample.get('no_rag_f1', 0.0)
        no_rag_em = sample.get('no_rag_em', 0.0)
        naive_rag_f1 = sample.get('naive_rag_f1', 0.0)
        naive_rag_em = sample.get('naive_rag_em', 0.0)
        
        # 获取检索时间
        naive_rag_retrieval_time = 0.0
        if question in naive_rag_results:
            naive_rag_retrieval_time = naive_rag_results[question].get('retrieval_time', 0.0)
        else:
            stats['missing_time'] += 1
        
        # 计算软标签 (含成本)
        soft_label, utility_gap, Q_no_rag, Q_naive_rag, U_no_rag, U_naive_rag, cost_norm = \
            compute_soft_label_with_cost(
                no_rag_f1, no_rag_em,
                naive_rag_f1, naive_rag_em,
                naive_rag_retrieval_time,
                max_retrieval_time,
                alpha=alpha,
                lambda_cost=lambda_cost,
                temperature=temperature,
            )
        
        # 构建新样本
        new_sample = sample.copy()
        new_sample['soft_label'] = round(soft_label, 6)
        new_sample['utility_gap'] = round(utility_gap, 6)
        new_sample['Q_no_rag'] = round(Q_no_rag, 4)
        new_sample['Q_naive_rag'] = round(Q_naive_rag, 4)
        new_sample['U_no_rag'] = round(U_no_rag, 4)
        new_sample['U_naive_rag'] = round(U_naive_rag, 4)
        new_sample['cost_norm'] = round(cost_norm, 4)
        new_sample['retrieval_time'] = round(naive_rag_retrieval_time, 4)
        
        # 统计
        stats['total'] += 1
        sum_utility_gap += utility_gap
        sum_cost_norm += cost_norm
        
        if soft_label < 0.3:
            stats['soft_label_near_0'] += 1
        elif soft_label > 0.7:
            stats['soft_label_near_1'] += 1
        else:
            stats['soft_label_near_05'] += 1
        
        processed_samples.append(new_sample)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"软标签分布统计 (λ = {lambda_cost}):")
    print(f"{'='*60}")
    print(f"  总样本数: {stats['total']}")
    print(f"  缺少时间信息: {stats['missing_time']}")
    print(f"  最大检索时间: {max_retrieval_time:.4f} 秒")
    print(f"  平均归一化成本: {sum_cost_norm/stats['total']:.4f}")
    print(f"  平均 Utility Gap: {sum_utility_gap/stats['total']:.4f}")
    print(f"\n  软标签分布:")
    print(f"    soft_label < 0.3 (倾向 no_rag): {stats['soft_label_near_0']} ({stats['soft_label_near_0']/stats['total']*100:.1f}%)")
    print(f"    0.3 <= soft_label <= 0.7 (模糊): {stats['soft_label_near_05']} ({stats['soft_label_near_05']/stats['total']*100:.1f}%)")
    print(f"    soft_label > 0.7 (倾向 naive_rag): {stats['soft_label_near_1']} ({stats['soft_label_near_1']/stats['total']*100:.1f}%)")
    print(f"{'='*60}")
    
    # 保存
    output_data = {"samples": processed_samples}
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存到: {output_file}")
    
    return processed_samples


def main():
    parser = argparse.ArgumentParser(description='计算软标签 (含成本考量)')
    parser.add_argument('--labels', type=str, 
                        default='D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/label_analysis/all_labels.json',
                        help='输入的 all_labels.json 文件')
    parser.add_argument('--naive_rag_results', type=str,
                        default='D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/Naiverag_results_20260127_012305.json',
                        help='naive_rag 结果文件 (包含 retrieval_time)')
    parser.add_argument('--output', type=str,
                        default='D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/label_analysis/all_labels_soft_with_cost.json',
                        help='输出文件')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='F1 权重 (默认 0.8, 即 Q = 0.8*F1 + 0.2*EM)')
    parser.add_argument('--lambda_cost', type=float, default=0.5,
                        help='成本权重 (默认 0.5)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='温度参数 (越小越接近硬标签, 默认 0.1)')
    
    args = parser.parse_args()
    
    process_labels_with_cost(
        labels_file=args.labels,
        naive_rag_results_file=args.naive_rag_results,
        output_file=args.output,
        alpha=args.alpha,
        lambda_cost=args.lambda_cost,
        temperature=args.temperature,
    )


if __name__ == '__main__':
    main()
