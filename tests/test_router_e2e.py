#!/usr/bin/env python3
"""
端到端路由器测试脚本

测试训练好的路由器在完整 RAG 流程中的性能：
1. 加载训练好的 Router
2. 对测试集进行 routing 决策
3. 根据 routing 选择执行对应的 RAG 策略
4. 评估最终答案质量（EM, F1）和成本指标

使用方法:
    python test_router_e2e.py --model_path router_models/train_dc_em_f1/final \
                          --test_data HotpotQA_train_data \
                          --output results/router_e2e_test.json
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.trainable_router.routers.dc_router import DCRouter
from rag_implementations.naive_rag.naive_rag import NaiveRAG
from rag_implementations.no_rag.no_rag import NoRAG
from rag_implementations.graph_rag.graph_rag import GraphRAG
from config.settings import Settings


def compute_em(gold_answer: List[str], prediction: str) -> float:
    """
    计算 Exact Match (EM) 分数
    
    Args:
        gold_answer: 标准答案列表
        prediction: 预测答案
        
    Returns:
        EM 分数 (0.0 或 1.0)
    """
    prediction = prediction.strip().lower()
    
    for ans in gold_answer:
        if ans.strip().lower() == prediction:
            return 1.0
    
    return 0.0


def compute_f1(gold_answer: List[str], prediction: str) -> float:
    """
    计算 F1 分数
    
    Args:
        gold_answer: 标准答案列表
        prediction: 预测答案
        
    Returns:
        F1 分数 (0.0-1.0)
    """
    def tokenize(s: str) -> List[str]:
        return s.lower().split()
    
    prediction_tokens = tokenize(prediction)
    
    if len(prediction_tokens) == 0:
        return 0.0
    
    # 尝试每个标准答案，取最高分
    best_f1 = 0.0
    
    for ans in gold_answer:
        gold_tokens = tokenize(ans)
        
        if len(gold_tokens) == 0:
            continue
        
        common_tokens = set(prediction_tokens) & set(gold_tokens)
        
        if len(common_tokens) == 0:
            continue
        
        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        best_f1 = max(best_f1, f1)
    
    return best_f1


def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """
    加载测试数据
    
    Args:
        data_path: 数据路径（文件或目录）
        
    Returns:
        测试数据列表
    """
    items = []
    
    if os.path.isdir(data_path):
        # 目录：加载所有 JSON 文件
        for filename in os.listdir(data_path):
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(data_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取 predictions
            if 'predictions' in data:
                items.extend(data['predictions'])
            else:
                # 单个 item 格式
                items.append(data)
    
    elif os.path.isfile(data_path):
        # 单个文件
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'predictions' in data:
            items = data['predictions']
        else:
            items = [data]
    
    return items


def run_e2e_test(
    router: DCRouter,
    rag_implementations: Dict[str, Any],
    test_data: List[Dict[str, Any]],
    max_samples: int = None,
    use_parallel: bool = False,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    运行端到端测试

    Args:
        router: 路由器
        rag_implementations: RAG 实现映射
        test_data: 测试数据
        max_samples: 最大测试样本数（None 表示全部）
        use_parallel: 是否使用并行处理
        num_workers: 并行工作线程数（仅在 use_parallel=True 时生效）

    Returns:
        测试结果列表
    """
    if max_samples:
        test_data = test_data[:max_samples]

    results = []

    def process_single_item(i, item, router, rag_implementations):
        """处理单个测试样本"""
        question = item.get('question', '')
        gold_answer = item.get('gold_answer', [])

        if not question:
            return None

        # 1. 路由决策
        predicted_strategy = router.route(question)

        # 2. 执行 RAG
        rag_impl = rag_implementations.get(predicted_strategy)

        if rag_impl is None:
            result = {
                'question': question,
                'gold_answer': gold_answer,
                'predicted_strategy': predicted_strategy,
                'answer': '',
                'em': 0.0,
                'f1': 0.0,
                'total_time': 0.0,
                'retrieval_time': 0.0,
                'generation_time': 0.0,
                'error': f"Strategy {predicted_strategy} not found",
            }
        else:
            try:
                # 执行 RAG
                answer = rag_impl.execute(question)

                # 计算评估指标
                em = compute_em(gold_answer, answer)
                f1 = compute_f1(gold_answer, answer)

                # 时间指标：直接从 RAG 实现获取
                # NoRAG: 只有 last_generation_time
                # NaiveRAG/GraphRAG: last_retrieval_time + last_generation_time
                if predicted_strategy == 'no_rag':
                    retrieval_time = 0.0
                    generation_time = rag_impl.last_generation_time
                    total_time = generation_time
                else:
                    retrieval_time = rag_impl.last_retrieval_time
                    generation_time = rag_impl.last_generation_time
                    total_time = retrieval_time + generation_time

                result = {
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_strategy': predicted_strategy,
                    'answer': answer,
                    'em': em,
                    'f1': f1,
                    'total_time': total_time,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                }

            except Exception as e:
                result = {
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_strategy': predicted_strategy,
                    'answer': '',
                    'em': 0.0,
                    'f1': 0.0,
                    'total_time': 0.0,
                    'retrieval_time': 0.0,
                    'generation_time': 0.0,
                    'error': str(e),
                }

        return result

    if use_parallel:
        # 并行处理模式
        print(f"使用并行处理（{num_workers} 个工作线程）...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_single_item, i, item, router, rag_implementations): i
                for i, item in enumerate(test_data)
            }

            # 收集结果
            for future in tqdm(as_completed(future_to_index), total=len(test_data), desc="处理测试样本"):
                result = future.result()
                if result:
                    results.append(result)

        # 按原始顺序排序结果
        index_to_result = {future_to_index[future]: future.result() for future in future_to_index}
        results = [index_to_result[i] for i in range(len(test_data)) if index_to_result.get(i) is not None]
    else:
        # 串行处理模式（原有逻辑）
        for i, item in enumerate(test_data):
            question = item.get('question', '')
            gold_answer = item.get('gold_answer', [])

            if not question:
                continue

            print(f"\n[{i+1}/{len(test_data)}] 测试问题: {question}")

            # 1. 路由决策
            predicted_strategy = router.route(question)
            print(f"  路由决策: {predicted_strategy}")

            # 2. 执行 RAG
            rag_impl = rag_implementations.get(predicted_strategy)

            if rag_impl is None:
                print(f"  错误：未找到策略 '{predicted_strategy}' 对应的 RAG 实现")
                result = {
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_strategy': predicted_strategy,
                    'answer': '',
                    'em': 0.0,
                    'f1': 0.0,
                    'total_time': 0.0,
                    'retrieval_time': 0.0,
                    'generation_time': 0.0,
                    'error': f"Strategy {predicted_strategy} not found",
                }
            else:
                try:
                    # 执行 RAG
                    answer = rag_impl.execute(question)

                    print(f"  预测答案: {answer}")

                    # 计算评估指标
                    em = compute_em(gold_answer, answer)
                    f1 = compute_f1(gold_answer, answer)

                    # 时间指标：直接从 RAG 实现获取
                    # NoRAG: 只有 last_generation_time
                    # NaiveRAG/GraphRAG: last_retrieval_time + last_generation_time
                    if predicted_strategy == 'no_rag':
                        retrieval_time = 0.0
                        generation_time = rag_impl.last_generation_time
                        total_time = generation_time
                    else:
                        retrieval_time = rag_impl.last_retrieval_time
                        generation_time = rag_impl.last_generation_time
                        total_time = retrieval_time + generation_time

                    result = {
                        'question': question,
                        'gold_answer': gold_answer,
                        'predicted_strategy': predicted_strategy,
                        'answer': answer,
                        'em': em,
                        'f1': f1,
                        'total_time': total_time,
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                    }

                    print(f"  EM: {em:.2f}, F1: {f1:.2f}, Total: {total_time:.2f}s (Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s)")

                except Exception as e:
                    print(f"  错误: {str(e)}")
                    result = {
                        'question': question,
                        'gold_answer': gold_answer,
                        'predicted_strategy': predicted_strategy,
                        'answer': '',
                        'em': 0.0,
                        'f1': 0.0,
                        'total_time': 0.0,
                        'retrieval_time': 0.0,
                        'generation_time': 0.0,
                        'error': str(e),
                    }

            results.append(result)

    return results


def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算聚合指标
    
    Args:
        results: 测试结果列表
        
    Returns:
        聚合指标字典
    """
    if not results:
        return {}
    
    # 整体指标
    total_em = sum(r['em'] for r in results if 'em' in r)
    total_f1 = sum(r['f1'] for r in results if 'f1' in r)
    total_time = sum(r['total_time'] for r in results if 'total_time' in r)
    total_retrieval_time = sum(r['retrieval_time'] for r in results if 'retrieval_time' in r)
    total_generation_time = sum(r['generation_time'] for r in results if 'generation_time' in r)
    
    num_valid = len(results)
    
    # 按策略分组统计
    strategy_stats = defaultdict(lambda: {
        'count': 0,
        'em': 0.0,
        'f1': 0.0,
        'total_time': 0.0,
    })
    
    for r in results:
        strategy = r.get('predicted_strategy', 'unknown')
        if strategy in strategy_stats:
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['em'] += r.get('em', 0.0)
            strategy_stats[strategy]['f1'] += r.get('f1', 0.0)
            strategy_stats[strategy]['total_time'] += r.get('total_time', 0.0)
    
    # 计算平均指标
    metrics = {
        'overall': {
            'num_samples': num_valid,
            'em': total_em / num_valid if num_valid > 0 else 0.0,
            'f1': total_f1 / num_valid if num_valid > 0 else 0.0,
            'avg_total_time': total_time / num_valid if num_valid > 0 else 0.0,
            'avg_retrieval_time': total_retrieval_time / num_valid if num_valid > 0 else 0.0,
            'avg_generation_time': total_generation_time / num_valid if num_valid > 0 else 0.0,
        },
        'by_strategy': {}
    }
    
    for strategy, stats in strategy_stats.items():
        count = stats['count']
        metrics['by_strategy'][strategy] = {
            'count': count,
            'ratio': count / num_valid if num_valid > 0 else 0.0,
            'em': stats['em'] / count if count > 0 else 0.0,
            'f1': stats['f1'] / count if count > 0 else 0.0,
            'avg_total_time': stats['total_time'] / count if count > 0 else 0.0,
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='端到端路由器测试')
    parser.add_argument('--model_path', type=str, required=True, help='Router 模型路径')
    parser.add_argument('--test_data', type=str, required=True, help='测试数据路径（文件或目录）')
    parser.add_argument('--output', type=str, default='', help='输出结果路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大测试样本数')
    parser.add_argument('--device', type=str, default='auto', help='Router 设备 (auto, cpu, cuda)')
    parser.add_argument('--config', type=str, default='config/settings.yaml', help='配置文件路径')
    parser.add_argument('--use_parallel', action='store_true', default=False, help='是否使用并行处理')
    parser.add_argument('--num_workers', type=int, default=4, help='并行工作线程数（仅在 use_parallel=True 时生效）')
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误：模型不存在: {args.model_path}")
        return
    
    # 加载配置
    print(f"加载配置: {args.config}")
    settings = Settings(args.config)
    
    # 初始化 Router
    print(f"\n加载 Router 模型: {args.model_path}")
    router = DCRouter(args.model_path)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Router 设备: {device}")
    router.to(device)
    
    # 初始化 RAG 实现
    print("\n初始化 RAG 实现...")
    rag_implementations = {}
    
    try:
        print("  - NoRAG")
        no_rag = NoRAG(settings)
        no_rag.build_index(None)  # NoRAG 不需要索引
        rag_implementations['no_rag'] = no_rag
    except Exception as e:
        print(f"  警告：NoRAG 初始化失败: {e}")
    
    try:
        print("  - NaiveRAG")
        naive_rag = NaiveRAG(settings)
        # 检查是否已有索引
        if os.path.exists(naive_rag.index_path):
            print(f"    加载已有索引: {naive_rag.index_path}")
            naive_rag.load_index()
        else:
            print(f"    构建索引: {naive_rag.index_path}")
            naive_rag.build_index()
        rag_implementations['naive_rag'] = naive_rag
    except Exception as e:
        print(f"  警告：NaiveRAG 初始化失败: {e}")
    
    try:
        print("  - GraphRAG")
        graph_rag = GraphRAG(settings)
        # GraphRAG 索引路径在 settings.graphrag.root_dir
        graph_rag.build_index_from_path(settings.graphrag.root_dir)
        rag_implementations['graph_rag'] = graph_rag
    except Exception as e:
        print(f"  警告：GraphRAG 初始化失败: {e}")
    
    if not rag_implementations:
        print("错误：没有可用的 RAG 实现")
        return
    
    # 加载测试数据
    print(f"\n加载测试数据: {args.test_data}")
    test_data = load_test_data(args.test_data)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"测试样本数: {len(test_data)} (限制为 {args.max_samples})")
    else:
        print(f"测试样本数: {len(test_data)}")
    
    # 运行端到端测试
    print("\n" + "=" * 60)
    if args.use_parallel:
        print(f"开始端到端测试（并行模式，{args.num_workers} 个工作线程）")
    else:
        print("开始端到端测试（串行模式）")
    print("=" * 60)

    start_time = datetime.now()
    results = run_e2e_test(
        router,
        rag_implementations,
        test_data,
        args.max_samples,
        use_parallel=args.use_parallel,
        num_workers=args.num_workers
    )
    end_time = datetime.now()
    
    # 计算聚合指标
    print("\n" + "=" * 60)
    print("计算聚合指标")
    print("=" * 60)
    
    metrics = compute_aggregate_metrics(results)
    
    # 打印结果
    print("\n整体性能:")
    overall = metrics['overall']
    print(f"  样本数: {overall['num_samples']}")
    print(f"  EM: {overall['em']:.4f}")
    print(f"  F1: {overall['f1']:.4f}")
    print(f"  平均总时间: {overall['avg_total_time']:.4f}s")
    print(f"  平均检索时间: {overall['avg_retrieval_time']:.4f}s")
    print(f"  平均生成时间: {overall['avg_generation_time']:.4f}s")
    
    print("\n各策略统计:")
    for strategy, stats in metrics['by_strategy'].items():
        print(f"  {strategy}:")
        print(f"    样本数: {stats['count']} ({stats['ratio']:.2%})")
        print(f"    EM: {stats['em']:.4f}")
        print(f"    F1: {stats['f1']:.4f}")
        print(f"    平均时间: {stats['avg_total_time']:.4f}s")
    
    # 保存结果
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        output_data = {
            'config': {
                'model_path': args.model_path,
                'test_data': args.test_data,
                'max_samples': args.max_samples,
                'device': device,
                'use_parallel': args.use_parallel,
                'num_workers': args.num_workers,
            },
            'metrics': metrics,
            'results': results,
            'test_time': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration': (end_time - start_time).total_seconds(),
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {args.output}")
    
    print("\n测试完成!")


if __name__ == '__main__':
    main()
