#!/usr/bin/env python3
"""
DPO偏好对数据生成脚本

将router label数据转换为DPO训练所需的偏好对格式。

二分类场景:
- no_rag (label=0) vs naive_rag (label=1)
- 根据optimal_strategy生成 (chosen, rejected) 对

三分类及以上场景（预留）:
- 根据最优策略生成所有可能的偏好对

用法示例:
    # 基本用法 - 二分类 (no_rag vs naive_rag)
    python generate_dpo_pairs.py \
        --input_file data/all_labels_no_tie.json \
        --output_file data/dpo_preference_pairs.json \
        --strategy_names no_rag naive_rag

    # 指定输入输出路径
    python scripts/data/generate_dpo_pairs.py \
        --input_file data/all_labels_no_tie.json \
        --output_file data/dpo/dpo_preference_pairs.json \
        --strategy_names no_rag naive_rag

    # 多分类模式（3个及以上策略）
    python generate_dpo_pairs.py \
        --input_file data/multi_strategy_labels.json \
        --output_file data/dpo_multiclass_pairs.json \
        --strategy_names no_rag naive_rag advanced_rag \
        --multiclass

输入文件格式:
    支持两种JSON格式:
    1. 完整格式（含scores）:
        {"samples": [
            {"question": "...", "optimal_strategy": "no_rag", 
             "no_rag_score": 0.8, "naive_rag_score": 0.5}
        ]}
    2. 简单格式（只有label）:
        {"samples": [
            {"question": "...", "optimal_strategy": "no_rag"}
        ]}

输出文件格式:
    DPO偏好对列表:
    [
        {
            "prompt": "问题文本",
            "chosen": 0,           // 偏好策略的索引
            "rejected": 1,         // 非偏好策略的索引
            "question_id": "q_0",
            "optimal_strategy": "no_rag",
            "strategy_names": ["no_rag", "naive_rag"]
        }
    ]

参数说明:
    --input_file, -i    输入的label数据文件路径（JSON格式，必需）
    --output_file, -o   输出的DPO偏好对文件路径（JSON格式，必需）
    --strategy_names    策略名称列表，默认: no_rag naive_rag
    --multiclass        使用多分类模式（3+策略）
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path


def load_label_data(file_path: str) -> Dict[str, Any]:
    """
    加载label数据文件
    
    支持两种格式:
    1. 完整格式（含scores）: {"samples": [{"question": "...", "optimal_strategy": "...", "no_rag_score": 0.5, ...}]}
    2. 简单格式（只有label）: {"samples": [{"question": "...", "optimal_strategy": "..."}]}
    
    Args:
        file_path: 输入文件路径
        
    Returns:
        解析后的数据字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果顶层是列表，包装成标准格式
    if isinstance(data, list):
        data = {"samples": data}
    
    return data


def generate_binary_preference_pairs(
    samples: List[Dict[str, Any]],
    strategy_names: List[str]
) -> List[Dict[str, Any]]:
    """
    生成二分类DPO偏好对
    
    对于二分类，每个样本只生成一个偏好对:
    - 如果optimal_strategy是strategy_names[0]，则 (0, 1)
    - 如果optimal_strategy是strategy_names[1]，则 (1, 0)
    
    Args:
        samples: 样本列表
        strategy_names: 策略名称列表（必须是2个）
        
    Returns:
        偏好对列表，每个元素包含:
        - prompt: 输入文本（问题）
        - chosen: 偏好的策略索引
        - rejected: 不喜欢的策略索引
        - question_id: 问题ID（如果有）
        - optimal_strategy: 最优策略名称
    """
    if len(strategy_names) != 2:
        raise ValueError(f"二分类需要恰好2个策略，当前有: {len(strategy_names)}")
    
    strategy_to_idx = {name: idx for idx, name in enumerate(strategy_names)}
    preference_pairs = []
    
    for idx, sample in enumerate(samples):
        question = sample.get('question', '')
        optimal_strategy = sample.get('optimal_strategy', '')
        
        if not question or not optimal_strategy:
            print(f"警告: 跳过第{idx}个样本，缺少question或optimal_strategy")
            continue
        
        # 处理tie情况：标记为no_rag (index 0)
        if optimal_strategy == 'tie':
            optimal_idx = 0  # no_rag
            other_idx = 1    # naive_rag
            optimal_strategy = 'no_rag'  # 更新标记
        elif optimal_strategy not in strategy_to_idx:
            print(f"警告: 跳过第{idx}个样本，未知的策略: {optimal_strategy}")
            continue
        else:
            # 确定chosen和rejected
            optimal_idx = strategy_to_idx[optimal_strategy]
            other_idx = 1 - optimal_idx  # 二分类中另一个策略的索引
        
        pair = {
            "prompt": question,
            "chosen": optimal_idx,
            "rejected": other_idx,
            "question_id": sample.get('question_id', f"q_{idx}"),
            "optimal_strategy": optimal_strategy,
            "strategy_names": strategy_names  # 保存策略名称映射
        }
        
        # 如果有详细分数，也保存下来（用于调试和分析）
        if 'no_rag_score' in sample or 'naive_rag_score' in sample:
            pair['scores'] = {
                name: sample.get(f'{name}_score', 0.0) 
                for name in strategy_names
            }
        
        preference_pairs.append(pair)
    
    return preference_pairs


def generate_multiclass_preference_pairs(
    samples: List[Dict[str, Any]],
    strategy_names: List[str]
) -> List[Dict[str, Any]]:
    """
    生成多分类DPO偏好对（预留，支持3+分类）
    
    参考EllieSQL的策略，对于N分类，从最优策略生成多个偏好对:
    - 最优策略 vs 其他每个策略
    - 如果score差异大，还可以生成次优策略vs更差策略的对
    
    Args:
        samples: 样本列表
        strategy_names: 策略名称列表（3个或以上）
        
    Returns:
        偏好对列表
    """
    if len(strategy_names) < 3:
        raise ValueError(f"多分类需要至少3个策略，当前有: {len(strategy_names)}")
    
    strategy_to_idx = {name: idx for idx, name in enumerate(strategy_names)}
    preference_pairs = []
    
    for idx, sample in enumerate(samples):
        question = sample.get('question', '')
        optimal_strategy = sample.get('optimal_strategy', '')
        
        if not question or optimal_strategy not in strategy_to_idx:
            continue
        
        optimal_idx = strategy_to_idx[optimal_strategy]
        
        # 策略1: 生成最优策略 vs 所有其他策略的偏好对
        for other_strategy in strategy_names:
            if other_strategy == optimal_strategy:
                continue
            other_idx = strategy_to_idx[other_strategy]
            
            pair = {
                "prompt": question,
                "chosen": optimal_idx,
                "rejected": other_idx,
                "question_id": sample.get('question_id', f"q_{idx}"),
                "optimal_strategy": optimal_strategy,
                "strategy_names": strategy_names,
                "pair_type": "optimal_vs_other"
            }
            preference_pairs.append(pair)
        
        # 策略2（可选）: 如果有score信息，生成次优vs更差的偏好对
        if 'no_rag_score' in sample or 'naive_rag_score' in sample:
            # 按score排序，生成相邻策略的偏好对
            scores = [
                (strategy_to_idx[name], sample.get(f'{name}_score', 0.0))
                for name in strategy_names
            ]
            scores.sort(key=lambda x: x[1], reverse=True)  # 按score降序
            
            # 生成相邻策略的偏好对
            for i in range(len(scores) - 1):
                better_idx, better_score = scores[i]
                worse_idx, worse_score = scores[i + 1]
                
                # 只生成score差异明显的对
                if better_score > worse_score:
                    pair = {
                        "prompt": question,
                        "chosen": better_idx,
                        "rejected": worse_idx,
                        "question_id": sample.get('question_id', f"q_{idx}"),
                        "optimal_strategy": optimal_strategy,
                        "strategy_names": strategy_names,
                        "pair_type": "ranked_pair",
                        "score_diff": better_score - worse_score
                    }
                    preference_pairs.append(pair)
    
    return preference_pairs


def save_preference_pairs(preference_pairs: List[Dict[str, Any]], output_file: str):
    """
    保存偏好对到JSON文件
    
    Args:
        preference_pairs: 偏好对列表
        output_file: 输出文件路径
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preference_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存 {len(preference_pairs)} 条偏好对到: {output_file}")


def print_statistics(preference_pairs: List[Dict[str, Any]], strategy_names: List[str]):
    """
    打印偏好对统计信息
    
    Args:
        preference_pairs: 偏好对列表
        strategy_names: 策略名称列表
    """
    print("\n" + "="*50)
    print("DPO偏好对数据统计")
    print("="*50)
    
    print(f"总偏好对数量: {len(preference_pairs)}")
    print(f"策略名称: {strategy_names}")
    
    if len(preference_pairs) == 0:
        return
    
    # 统计chosen分布
    chosen_counts = {}
    for pair in preference_pairs:
        chosen = pair['chosen']
        chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
    
    print("\nChosen分布（偏好策略）:")
    for idx, name in enumerate(strategy_names):
        count = chosen_counts.get(idx, 0)
        percentage = count / len(preference_pairs) * 100 if preference_pairs else 0
        print(f"  [{idx}] {name}: {count} ({percentage:.1f}%)")
    
    # 统计pair_type（如果有）
    if 'pair_type' in preference_pairs[0]:
        type_counts = {}
        for pair in preference_pairs:
            pt = pair.get('pair_type', 'unknown')
            type_counts[pt] = type_counts.get(pt, 0) + 1
        
        print("\n偏好对类型分布:")
        for pt, count in type_counts.items():
            percentage = count / len(preference_pairs) * 100
            print(f"  {pt}: {count} ({percentage:.1f}%)")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='将router label数据转换为DPO偏好对格式'
    )
    parser.add_argument(
        '--input_file', '-i',
        type=str,
        required=True,
        help='输入的label数据文件路径（JSON格式）'
    )
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        required=True,
        help='输出的DPO偏好对文件路径（JSON格式）'
    )
    parser.add_argument(
        '--strategy_names',
        type=str,
        nargs='+',
        default=['no_rag', 'naive_rag'],
        help='策略名称列表，默认为: no_rag naive_rag'
    )
    parser.add_argument(
        '--multiclass',
        action='store_true',
        help='使用多分类模式（3+策略），否则使用二分类模式'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.input_file}")
    data = load_label_data(args.input_file)
    samples = data.get('samples', [])
    print(f"找到 {len(samples)} 条样本")
    
    # 生成偏好对
    if args.multiclass or len(args.strategy_names) >= 3:
        print(f"使用多分类模式，策略: {args.strategy_names}")
        preference_pairs = generate_multiclass_preference_pairs(
            samples, args.strategy_names
        )
    else:
        print(f"使用二分类模式，策略: {args.strategy_names}")
        preference_pairs = generate_binary_preference_pairs(
            samples, args.strategy_names
        )
    
    # 打印统计
    print_statistics(preference_pairs, args.strategy_names)
    
    # 保存结果
    save_preference_pairs(preference_pairs, args.output_file)
    
    # 打印示例
    if preference_pairs:
        print("\n示例数据（前3条）:")
        for i, pair in enumerate(preference_pairs[:3]):
            print(f"\n[{i+1}] prompt: {pair['prompt'][:80]}...")
            print(f"    chosen: {pair['chosen']} ({args.strategy_names[pair['chosen']]})")
            print(f"    rejected: {pair['rejected']} ({args.strategy_names[pair['rejected']]})")


if __name__ == "__main__":
    main()
