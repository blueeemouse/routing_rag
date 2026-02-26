"""
数据处理脚本：过滤tie样本，准备训练数据

功能：
1. 从 all_labels.json 过滤掉 tie 样本
2. 支持分层采样
3. 生成 RouterLabelDataset 可用的格式
"""

import json
import os
import random
from collections import Counter
from typing import List, Dict, Any


def load_all_labels(file_path: str) -> List[Dict[str, Any]]:
    """加载 all_labels.json"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def filter_tie_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤掉 tie 样本"""
    filtered = [item for item in data if item.get('label') != 'tie']
    
    # 统计
    label_counts = Counter(item['label'] for item in filtered)
    print(f"过滤后样本数: {len(filtered)}")
    print(f"标签分布: {dict(label_counts)}")
    
    return filtered


def stratified_sample(
    data: List[Dict[str, Any]], 
    sample_size: int,
    label_key: str = 'label',
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    分层采样
    
    Args:
        data: 数据列表
        sample_size: 目标样本数
        label_key: 标签字段名
        seed: 随机种子
    
    Returns:
        采样后的数据列表
    """
    random.seed(seed)
    
    # 按标签分组
    label_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        label = item.get(label_key)
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    # 计算每个标签的采样数量
    total = len(data)
    sampled = []
    label_sample_counts = {}
    
    for label, items in label_groups.items():
        # 按比例分配
        ratio = len(items) / total
        n_samples = max(1, int(sample_size * ratio))
        # 不超过该标签的实际数量
        n_samples = min(n_samples, len(items))
        label_sample_counts[label] = n_samples
    
    # 补齐：如果总数不够，从最大类补齐
    current_total = sum(label_sample_counts.values())
    if current_total < sample_size:
        # 找到数量最多的类别，补齐差值
        max_label = max(label_groups.keys(), key=lambda l: len(label_groups[l]))
        deficit = sample_size - current_total
        # 确保不超过该类别的实际数量
        label_sample_counts[max_label] = min(
            label_sample_counts[max_label] + deficit,
            len(label_groups[max_label])
        )
    
    # 执行采样
    for label, n_samples in label_sample_counts.items():
        items = label_groups[label]
        sampled_items = random.sample(items, n_samples)
        sampled.extend(sampled_items)
        print(f"  {label}: 采样 {n_samples} 条 (共 {len(items)} 条)")
    
    # 打乱顺序
    random.shuffle(sampled)
    
    return sampled


def convert_to_router_label_format(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    转换为 RouterLabelDataset 格式
    
    格式:
    {
        "samples": [
            {
                "question": "...",
                "optimal_strategy": "no_rag" or "naive_rag"
            }
        ]
    }
    """
    samples = []
    for item in data:
        label = item.get('label')
        # label 是 'no_rag' 或 'naive_rag'
        samples.append({
            'question': item['question'],
            'optimal_strategy': label
        })
    
    return {'samples': samples}


def main():
    """主函数"""
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(base_dir, 'HotpotQA_train_data', 'label_analysis', 'all_labels.json')
    output_dir = os.path.join(base_dir, 'HotpotQA_train_data', 'label_analysis')
    
    # 配置
    sample_size = 1000  # None 表示不采样，使用全部数据；设为整数表示采样数量
    
    print("=" * 60)
    print("数据处理：过滤 tie 样本")
    print("=" * 60)
    
    # 1. 加载数据
    print(f"\n加载数据: {input_path}")
    data = load_all_labels(input_path)
    print(f"原始样本数: {len(data)}")
    
    # 原始标签分布
    original_dist = Counter(item['label'] for item in data)
    print(f"原始标签分布: {dict(original_dist)}")
    
    # 2. 过滤 tie 样本
    print("\n过滤 tie 样本...")
    filtered_data = filter_tie_samples(data)
    
    # 3. 可选：采样
    if sample_size is not None:
        print(f"\n分层采样 {sample_size} 条...")
        final_data = stratified_sample(filtered_data, sample_size)
        output_suffix = f"_sampled{sample_size}"
    else:
        final_data = filtered_data
        output_suffix = ""
    
    # 4. 转换格式
    print("\n转换格式...")
    output_data = convert_to_router_label_format(final_data)
    
    # 5. 保存
    # 文件名包含 'all_labels' 以触发 RouterLabelDataset（根据 train_router.py 的判断逻辑）
    output_filename = f"all_labels_no_tie{output_suffix}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存到: {output_path}")
    print(f"最终样本数: {len(output_data['samples'])}")
    
    # 最终标签分布
    final_dist = Counter(s['optimal_strategy'] for s in output_data['samples'])
    print(f"最终标签分布: {dict(final_dist)}")
    
    # 计算建议的类别权重
    print("\n" + "=" * 60)
    print("建议的类别权重（用于交叉熵损失）")
    print("=" * 60)
    total = sum(final_dist.values())
    
    print("\n方式1: 逆频率权重")
    inv_weights = {}
    for label, count in final_dist.items():
        weight = total / count
        inv_weights[label] = weight
        print(f"  {label}: {weight:.2f} (样本数: {count})")
    
    print("\n方式2: 平方根逆频率权重（更温和，推荐）")
    sqrt_weights = {}
    import math
    for label, count in final_dist.items():
        weight = math.sqrt(total / count)
        sqrt_weights[label] = weight
        print(f"  {label}: {weight:.2f} (样本数: {count})")
    
    # 推荐权重配置
    print("\n推荐命令行参数（平方根逆频率）:")
    weights_str = ",".join(f"{label}={sqrt_weights[label]:.1f}" for label in sorted(sqrt_weights.keys()))
    print(f"  --class_weights \"{weights_str}\"")


if __name__ == '__main__':
    main()
