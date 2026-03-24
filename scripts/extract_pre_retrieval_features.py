#!/usr/bin/env python3
"""
Pre-Retrieval 特征预提取脚本

功能：
- 遍历训练数据集
- 对每个 query 进行试探性检索（top-k=3，使用真实 NaiveRAG 索引）
- 提取 Pre-Retrieval 特征
- 保存为 JSON 文件

用法:
    # 提取训练数据特征
    python scripts/extract_pre_retrieval_features.py \\
        --data_path HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json \\
        --output_path pre_retrieval_features/train_features.json \\
        --index_path naive_rag_index_hotpotqa_train_5000_samples
    
    # 提取测试数据特征
    python scripts/extract_pre_retrieval_features.py \\
        --data_path evaluation_results/router_test_labels.json \\
        --output_path pre_retrieval_features/test_features.json \\
        --index_path naive_rag_index_hotpotqa_1000_samples
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.trainable_router.feature_extraction import PreRetrievalFeatureExtractor


def load_training_data(data_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    加载训练数据
    
    Args:
        data_path: 数据文件路径
        limit: 限制加载的样本数量（用于测试）
    
    Returns:
        样本列表
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    
    if limit:
        samples = samples[:limit]
    
    return samples


def create_naive_rag_retriever(index_path: str, top_k: int = 3):
    """
    创建真实的 NaiveRAG 检索器
    
    Args:
        index_path: 索引路径
        top_k: 检索返回的文档数量
    
    Returns:
        检索器对象或 None
    """
    try:
        from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
        
        # 创建 NaiveRAG 实例
        naive_rag = NaiveRAG()
        
        # 加载索引
        print(f"加载索引：{index_path}")
        success = naive_rag.load_index(index_path)
        if not success:
            print(f"❌ 加载索引失败：{index_path}")
            return None
        
        # 创建检索器
        if naive_rag.is_index_initialized:
            retriever = naive_rag.index.as_retriever(similarity_top_k=top_k)
            print(f"✓ NaiveRAG 检索器创建成功（top_k={top_k}）")
            return retriever
        else:
            print("❌ 索引未初始化")
            return None
            
    except Exception as e:
        print(f"❌ 创建 NaiveRAG 检索器失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features_for_sample(
    query: str, 
    retriever: Any, 
    extractor: PreRetrievalFeatureExtractor
) -> Dict[str, float]:
    """
    为单个样本提取 Pre-Retrieval 特征
    
    Args:
        query: 查询字符串
        retriever: 检索器
        extractor: 特征提取器
    
    Returns:
        特征字典
    """
    try:
        # 执行检索
        nodes = retriever.retrieve(query)
        
        # 提取特征
        features = extractor.extract_from_nodes(nodes)
        
        return features
    
    except Exception as e:
        print(f"  ⚠ 提取特征失败：{e}")
        return extractor._zero_features()


def extract_all_features(
    samples: List[Dict[str, Any]], 
    retriever: Any,
    output_path: str,
    batch_size: int = 50,
    save_interval: int = 200
):
    """
    批量提取所有样本的特征
    
    Args:
        samples: 样本列表
        retriever: 检索器
        output_path: 输出文件路径
        batch_size: 批处理大小
        save_interval: 保存间隔
    """
    extractor = PreRetrievalFeatureExtractor(similarity_threshold=0.5)
    
    results = []
    total = len(samples)
    
    print(f"开始提取特征，共 {total} 个样本...")
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        query = sample.get('question', '')
        
        # 提取特征
        features = extract_features_for_sample(query, retriever, extractor)
        
        # 保存结果
        result = {
            'question': query,
            'optimal_strategy': sample.get('optimal_strategy', 'unknown'),
            'no_rag_score': sample.get('no_rag_score', 0.0),
            'naive_rag_score': sample.get('naive_rag_score', 0.0),
            'pre_features': [features[name] for name in extractor.feature_names],
            'feature_names': extractor.feature_names,
        }
        results.append(result)
        
        # 进度更新
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (total - i - 1) * avg_time
            
            print(f"  进度：{i+1}/{total} ({(i+1)/total*100:.1f}%), "
                  f"平均耗时：{avg_time:.3f}s/样本，"
                  f"剩余时间：{remaining:.1f}s")
        
        # 定期保存（防止意外丢失）
        if (i + 1) % save_interval == 0:
            save_intermediate(results, output_path + '.tmp')
            print(f"  ✓ 已保存临时文件：{output_path}.tmp")
    
    # 保存最终结果
    save_results(results, output_path)
    
    total_time = time.time() - start_time
    print(f"\n✓ 特征提取完成！")
    print(f"  总耗时：{total_time:.1f}s")
    print(f"  平均速度：{total/total_time:.2f} 样本/秒")
    print(f"  输出文件：{output_path}")


def save_intermediate(results: List[Dict], path: str):
    """保存中间结果"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'samples': results, 'partial': True}, f, indent=2, ensure_ascii=False)


def save_results(results: List[Dict], path: str):
    """保存最终结果"""
    # 统计特征分布
    feature_names = results[0]['feature_names'] if results else []
    feature_stats = {}

    for feat_idx, feat_name in enumerate(feature_names):
        values = [r['pre_features'][feat_idx] for r in results]
        feature_stats[feat_name] = {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
        }
    
    output_data = {
        'samples': results,
        'total': len(results),
        'feature_names': feature_names,
        'feature_stats': feature_stats,
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 结果已保存到：{path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Pre-Retrieval 特征预提取脚本')
    
    # 数据配置
    parser.add_argument('--data_path', type=str, required=True, 
                        help='训练数据文件路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出文件路径')
    
    # 检索配置
    parser.add_argument('--index_path', type=str, required=True,
                        help='NaiveRAG 索引路径')
    parser.add_argument('--top_k', type=int, default=3,
                        help='检索返回的文档数量')
    
    # 批处理配置
    parser.add_argument('--batch_size', type=int, default=50,
                        help='批处理大小（进度更新间隔）')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='保存间隔（每多少个样本保存一次）')
    
    # 测试配置
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理的样本数量（用于测试）')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载训练数据：{args.data_path}")
    samples = load_training_data(args.data_path, limit=args.limit)
    print(f"✓ 加载了 {len(samples)} 个样本")
    
    # 创建检索器
    print(f"创建检索器（top_k={args.top_k}）...")
    retriever = create_naive_rag_retriever(args.index_path, args.top_k)
    
    if retriever is None:
        print("❌ 无法创建检索器，退出")
        sys.exit(1)
    
    # 提取特征
    extract_all_features(
        samples=samples,
        retriever=retriever,
        output_path=args.output_path,
        batch_size=args.batch_size,
        save_interval=args.save_interval
    )


if __name__ == '__main__':
    main()
