"""
计算软标签

软标签公式:
- Q = alpha * F1 + (1-alpha) * EM  (质量分数, 默认 alpha=0.8)
- ΔU = Q_naive_rag - Q_no_rag
- soft_label = sigmoid(ΔU / τ) = 1 / (1 + exp(-ΔU / τ))

soft_label 含义:
- 接近 0: no_rag 更好
- 接近 0.5: tie (两者差不多)
- 接近 1: naive_rag 更好
"""

import json
import math
import argparse
from pathlib import Path


def compute_soft_label(
    no_rag_f1: float,
    no_rag_em: float,
    naive_rag_f1: float,
    naive_rag_em: float,
    alpha: float = 0.8,
    temperature: float = 0.1,
) -> tuple[float, float]:
    """
    计算软标签
    
    Args:
        no_rag_f1, no_rag_em: no_rag 的 F1 和 EM 分数
        naive_rag_f1, naive_rag_em: naive_rag 的 F1 和 EM 分数
        alpha: F1 权重 (默认 0.8, 即 Q = 0.8*F1 + 0.2*EM)
        temperature: 温度参数 (越小越接近硬标签)
    
    Returns:
        (soft_label, utility_gap)
        - soft_label: 0~1 的软标签值
        - utility_gap: ΔU 值
    """
    # 质量分数
    Q_no_rag = alpha * no_rag_f1 + (1 - alpha) * no_rag_em
    Q_naive_rag = alpha * naive_rag_f1 + (1 - alpha) * naive_rag_em
    
    # Utility gap (naive_rag 相对于 no_rag 的优势)
    utility_gap = Q_naive_rag - Q_no_rag
    
    # 软标签 (sigmoid)
    if temperature <= 0:
        temperature = 0.001  # 防止除零
    
    soft_label = 1.0 / (1.0 + math.exp(-utility_gap / temperature))
    
    return soft_label, utility_gap


def process_labels(
    input_file: str,
    output_file: str,
    alpha: float = 0.8,
    temperature: float = 0.1,
    remove_tie: bool = False,
):
    """
    处理标签文件, 添加软标签
    
    Args:
        input_file: 输入的 all_labels.json 文件
        output_file: 输出文件
        alpha: F1 权重
        temperature: 温度参数
        remove_tie: 是否移除 tie 样本 (目前软标签方案不推荐移除)
    """
    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式: {"samples": [...]} 或 [...]
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    else:
        samples = data
    
    print(f"共 {len(samples)} 条样本")
    
    # 处理每条样本
    processed_samples = []
    stats = {
        'total': 0,
        'soft_label_near_0': 0,  # < 0.3
        'soft_label_near_05': 0,  # 0.3 ~ 0.7
        'soft_label_near_1': 0,  # > 0.7
    }
    
    for sample in samples:
        # 兼容不同字段名
        no_rag_f1 = sample.get('no_rag_f1', 0.0)
        no_rag_em = sample.get('no_rag_em', 0.0)
        naive_rag_f1 = sample.get('naive_rag_f1', 0.0)
        naive_rag_em = sample.get('naive_rag_em', 0.0)
        
        # 计算软标签
        soft_label, utility_gap = compute_soft_label(
            no_rag_f1, no_rag_em,
            naive_rag_f1, naive_rag_em,
            alpha=alpha,
            temperature=temperature,
        )
        
        # 构建新样本
        new_sample = sample.copy()
        new_sample['soft_label'] = round(soft_label, 6)
        new_sample['utility_gap'] = round(utility_gap, 6)
        
        # 统计
        stats['total'] += 1
        if soft_label < 0.3:
            stats['soft_label_near_0'] += 1
        elif soft_label > 0.7:
            stats['soft_label_near_1'] += 1
        else:
            stats['soft_label_near_05'] += 1
        
        processed_samples.append(new_sample)
    
    # 打印统计信息
    print(f"\n软标签分布统计:")
    print(f"  总样本数: {stats['total']}")
    print(f"  soft_label < 0.3 (倾向 no_rag): {stats['soft_label_near_0']} ({stats['soft_label_near_0']/stats['total']*100:.1f}%)")
    print(f"  0.3 <= soft_label <= 0.7 (模糊): {stats['soft_label_near_05']} ({stats['soft_label_near_05']/stats['total']*100:.1f}%)")
    print(f"  soft_label > 0.7 (倾向 naive_rag): {stats['soft_label_near_1']} ({stats['soft_label_near_1']/stats['total']*100:.1f}%)")
    
    # 保存
    output_data = {"samples": processed_samples}
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存到: {output_file}")
    
    return processed_samples


def main():
    parser = argparse.ArgumentParser(description='计算软标签')
    parser.add_argument('--input', type=str, 
                        default='D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/label_analysis/all_labels.json',
                        help='输入的 all_labels.json 文件')
    parser.add_argument('--output', type=str,
                        default='D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/label_analysis/all_labels_soft.json',
                        help='输出文件')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='F1 权重 (默认 0.8, 即 Q = 0.8*F1 + 0.2*EM)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='温度参数 (越小越接近硬标签, 默认 0.1)')
    
    args = parser.parse_args()
    
    process_labels(
        input_file=args.input,
        output_file=args.output,
        alpha=args.alpha,
        temperature=args.temperature,
    )


if __name__ == '__main__':
    main()
