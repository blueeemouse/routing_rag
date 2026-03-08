#!/usr/bin/env python3
"""
Oracle Routing 测试脚本

使用 Oracle（ground truth）标签来决定每个问题应该使用哪个 RAG 策略，
以评估在完美路由决策下的最优性能。

这可以帮助我们了解当前 RAG 系统的上限性能，以及不同策略组合的潜力。

使用方法:
    python test_oracle_routing.py \
        --hotpotqa_file HotpotQA/hotpot_dev_distractor_1000_samples.jsonl \
        --oracle_labels_file evaluation_results/router_test_labels.json \
        --naive_rag_index_path naive_rag_index_hotpotqa_1000_samples \
        --output results/oracle_routing_eval.json
    
    # 使用并行处理加速
    python test_oracle_routing.py \
        --hotpotqa_file HotpotQA/hotpot_dev_distractor_1000_samples.jsonl \
        --oracle_labels_file evaluation_results/router_test_labels.json \
        --naive_rag_index_path naive_rag_index_hotpotqa_1000_samples \
        --output results/oracle_routing_eval.json \
        --use_parallel --num_workers 4
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

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from rag_implementations.no_rag.no_rag_impl import NoRAG
from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
from config.config import settings


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


def load_test_data(data_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """
    加载HotpotQA JSONL格式的测试数据
    
    Args:
        data_path: HotpotQA JSONL文件路径
        max_samples: 最大加载样本数（None 表示全部）
        
    Returns:
        测试数据列表，每个元素包含：
        - question: 问题文本
        - gold_answer: 标准答案列表
    """
    items = []
    
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在: {data_path}")
        return items
    
    # 加载 JSONL 文件
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                
                # 提取标准答案（answer + answer_aliases）
                gold_answers = [data['answer']]
                if 'answer_aliases' in data:
                    gold_answers.extend(data['answer_aliases'])
                
                items.append({
                    'question': data['question'],
                    'gold_answer': gold_answers
                })
            except Exception as e:
                print(f"警告：跳过第 {i+1} 行，解析错误: {e}")
                continue
    
    return items


def load_oracle_labels(labels_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载 Oracle 标签文件
    
    Args:
        labels_path: Oracle 标签 JSON 文件路径
        
    Returns:
        问题到标签信息的映射字典
    """
    if not os.path.exists(labels_path):
        print(f"错误：标签文件不存在: {labels_path}")
        return {}
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    question_to_label = {}
    for sample in data.get('samples', []):
        question = sample.get('question', '')
        if question:
            question_to_label[question] = {
                'optimal_strategy': sample.get('optimal_strategy', 'no_rag'),
                'no_rag_score': sample.get('no_rag_score', 0.0),
                'naive_rag_score': sample.get('naive_rag_score', 0.0),
                'score_diff': sample.get('score_diff', 0.0),
            }
    
    print(f"成功加载 {len(question_to_label)} 个 Oracle 标签")
    return question_to_label


def run_oracle_test(
    rag_implementations: Dict[str, Any],
    test_data: List[Dict[str, Any]],
    oracle_labels: Dict[str, Dict[str, Any]],
    max_samples: int = None,
    use_parallel: bool = False,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    运行 Oracle 路由测试

    Args:
        rag_implementations: RAG 实现映射
        test_data: 测试数据
        oracle_labels: Oracle 标签映射
        max_samples: 最大测试样本数（None 表示全部）
        use_parallel: 是否使用并行处理
        num_workers: 并行工作线程数（仅在 use_parallel=True 时生效）

    Returns:
        测试结果列表
    """
    if max_samples:
        test_data = test_data[:max_samples]

    results = []

    def process_single_item(i, item, rag_implementations, oracle_labels):
        """处理单个测试样本"""
        question = item.get('question', '')
        gold_answer = item.get('gold_answer', [])

        if not question:
            return None

        # 1. 使用 Oracle 标签获取最优策略
        label_info = oracle_labels.get(question, {})
        predicted_strategy = label_info.get('optimal_strategy', 'no_rag')
        
        # 2. 执行 RAG
        rag_impl = rag_implementations.get(predicted_strategy)

        if rag_impl is None:
            result = {
                'question': question,
                'gold_answer': gold_answer,
                'strategy': predicted_strategy,
                'oracle_strategy': predicted_strategy,
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
                    'strategy': predicted_strategy,
                    'oracle_strategy': predicted_strategy,
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
                    'strategy': predicted_strategy,
                    'oracle_strategy': predicted_strategy,
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
                executor.submit(process_single_item, i, item, rag_implementations, oracle_labels): i
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
        # 串行处理模式
        for i, item in enumerate(test_data):
            question = item.get('question', '')
            gold_answer = item.get('gold_answer', [])

            if not question:
                continue

            print(f"\n[{i+1}/{len(test_data)}] 测试问题: {question}")

            # 1. 使用 Oracle 标签获取最优策略
            label_info = oracle_labels.get(question, {})
            predicted_strategy = label_info.get('optimal_strategy', 'no_rag')
            print(f"  Oracle 策略: {predicted_strategy}")

            # 2. 执行 RAG
            rag_impl = rag_implementations.get(predicted_strategy)

            if rag_impl is None:
                print(f"  错误：未找到策略 '{predicted_strategy}' 对应的 RAG 实现")
                result = {
                    'question': question,
                    'gold_answer': gold_answer,
                    'strategy': predicted_strategy,
                    'oracle_strategy': predicted_strategy,
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
                        'strategy': predicted_strategy,
                        'oracle_strategy': predicted_strategy,
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
                        'strategy': predicted_strategy,
                        'oracle_strategy': predicted_strategy,
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
        'retrieval_time': 0.0,
        'generation_time': 0.0,
    })
    
    for r in results:
        strategy = r.get('strategy', 'unknown')
        if strategy in strategy_stats:
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['em'] += r.get('em', 0.0)
            strategy_stats[strategy]['f1'] += r.get('f1', 0.0)
            strategy_stats[strategy]['total_time'] += r.get('total_time', 0.0)
            strategy_stats[strategy]['retrieval_time'] += r.get('retrieval_time', 0.0)
            strategy_stats[strategy]['generation_time'] += r.get('generation_time', 0.0)
    
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
            'avg_retrieval_time': stats['retrieval_time'] / count if count > 0 else 0.0,
            'avg_generation_time': stats['generation_time'] / count if count > 0 else 0.0,
        }
    
    return metrics


def print_comparison_results(oracle_labels: Dict[str, Dict[str, Any]], results: List[Dict[str, Any]]):
    """
    打印 Oracle 标签中的预计算分数与实际运行结果的对比
    """
    print("\n" + "=" * 60)
    print("Oracle 标签预计算分数 vs 实际运行结果对比")
    print("=" * 60)
    
    # 统计对比
    oracle_no_rag_better = sum(1 for q, label in oracle_labels.items() if label.get('optimal_strategy') == 'no_rag')
    oracle_naive_rag_better = sum(1 for q, label in oracle_labels.items() if label.get('optimal_strategy') == 'naive_rag')
    
    actual_no_rag_count = sum(1 for r in results if r.get('strategy') == 'no_rag')
    actual_naive_rag_count = sum(1 for r in results if r.get('strategy') == 'naive_rag')
    
    print(f"\nOracle 标签分布:")
    print(f"  no_rag: {oracle_no_rag_better} ({oracle_no_rag_better/len(oracle_labels)*100:.1f}%)")
    print(f"  naive_rag: {oracle_naive_rag_better} ({oracle_naive_rag_better/len(oracle_labels)*100:.1f}%)")
    
    print(f"\n实际执行分布:")
    print(f"  no_rag: {actual_no_rag_count} ({actual_no_rag_count/len(results)*100:.1f}%)")
    print(f"  naive_rag: {actual_naive_rag_count} ({actual_naive_rag_count/len(results)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Oracle Routing 测试')
    parser.add_argument('--hotpotqa_file', type=str, required=True, 
                        help='HotpotQA JSONL文件路径')
    parser.add_argument('--oracle_labels_file', type=str, required=True, 
                        help='Oracle 标签 JSON 文件路径')
    parser.add_argument('--naive_rag_index_path', type=str, required=True, 
                        help='NaiveRAG索引路径')
    parser.add_argument('--use_graphrag', action='store_true', default=False, 
                        help='是否使用GraphRAG（默认不使用）')
    parser.add_argument('--graphrag_work_dir', type=str, default='', 
                        help='GraphRAG工作目录（仅在use_graphrag=True时需要）')
    parser.add_argument('--output', type=str, default='', help='输出结果路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大测试样本数')
    parser.add_argument('--config', type=str, default='config/settings.yaml', help='配置文件路径')
    parser.add_argument('--use_parallel', action='store_true', default=False, help='是否使用并行处理')
    parser.add_argument('--num_workers', type=int, default=4, help='并行工作线程数')
    
    args = parser.parse_args()
    
    # 配置已通过全局settings加载
    print(f"使用配置: config/settings.yaml")
    
    # 加载 Oracle 标签
    print(f"\n加载 Oracle 标签: {args.oracle_labels_file}")
    oracle_labels = load_oracle_labels(args.oracle_labels_file)
    
    if not oracle_labels:
        print("错误：无法加载 Oracle 标签")
        return
    
    # 初始化 RAG 实现
    print("\n初始化 RAG 实现...")
    rag_implementations = {}

    # 初始化 NoRAG
    try:
        print("  - NoRAG")
        no_rag = NoRAG()
        rag_implementations['no_rag'] = no_rag
        print("    NoRAG 初始化成功")
    except Exception as e:
        print(f"    警告：NoRAG 初始化失败: {e}")

    # 初始化 NaiveRAG
    try:
        print("  - NaiveRAG")
        naive_rag = NaiveRAG()

        if os.path.exists(args.naive_rag_index_path):
            print(f"    加载索引: {args.naive_rag_index_path}")
            naive_rag.load_index(args.naive_rag_index_path)
            rag_implementations['naive_rag'] = naive_rag
            print("    NaiveRAG 初始化成功")
        else:
            print(f"    错误：索引路径不存在: {args.naive_rag_index_path}")
    except Exception as e:
        print(f"    警告：NaiveRAG 初始化失败: {e}")
    
    # 初始化 GraphRAG（可选）
    if args.use_graphrag:
        try:
            print("  - GraphRAG")
            graph_rag = GraphRAG()

            if not args.graphrag_work_dir:
                print("    错误：未指定GraphRAG工作目录")
            else:
                print(f"    加载GraphRAG索引: {args.graphrag_work_dir}")
                graph_rag.build_index_from_path(args.graphrag_work_dir)
                rag_implementations['graph_rag'] = graph_rag
                print("    GraphRAG 初始化成功")
        except Exception as e:
            print(f"    警告：GraphRAG 初始化失败: {e}")
    
    if not rag_implementations:
        print("错误：没有可用的 RAG 实现")
        return
    
    print(f"\n成功初始化 {len(rag_implementations)} 个 RAG 实现: {list(rag_implementations.keys())}")
    
    # 加载测试数据
    print(f"\n加载测试数据: {args.hotpotqa_file}")
    test_data = load_test_data(args.hotpotqa_file, args.max_samples)
    
    if not test_data:
        print("错误：未加载到任何测试数据")
        return
    
    print(f"成功加载 {len(test_data)} 个测试样本")
    
    # 运行 Oracle 测试
    print("\n" + "=" * 60)
    print("开始 Oracle Routing 测试")
    print("=" * 60)
    if args.use_parallel:
        print(f"并行模式，{args.num_workers} 个工作线程")
    else:
        print("串行模式")

    start_time = datetime.now()
    results = run_oracle_test(
        rag_implementations,
        test_data,
        oracle_labels,
        args.max_samples,
        use_parallel=args.use_parallel,
        num_workers=args.num_workers
    )
    end_time = datetime.now()
    
    # 打印对比结果
    print_comparison_results(oracle_labels, results)
    
    # 计算聚合指标
    print("\n" + "=" * 60)
    print("计算聚合指标")
    print("=" * 60)
    
    metrics = compute_aggregate_metrics(results)
    
    # 打印结果
    print("\n整体性能 (Oracle Routing):")
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
        print(f"    平均总时间: {stats['avg_total_time']:.4f}s")
        print(f"    平均检索时间: {stats['avg_retrieval_time']:.4f}s")
        print(f"    平均生成时间: {stats['avg_generation_time']:.4f}s")
    
    # 保存结果
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        output_data = {
            'config': {
                'hotpotqa_file': args.hotpotqa_file,
                'oracle_labels_file': args.oracle_labels_file,
                'naive_rag_index_path': args.naive_rag_index_path,
                'use_graphrag': args.use_graphrag,
                'graphrag_work_dir': args.graphrag_work_dir if args.use_graphrag else None,
                'max_samples': args.max_samples,
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
    
    print("\nOracle Routing 测试完成!")


if __name__ == '__main__':
    main()
