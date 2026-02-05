"""
测试 Router 模型的路由决策
使用训练好的 Router 模型在 HotpotQA 测试集上进行路由选择

支持计算准确率（如果提供策略结果文件）
"""

import json
import sys
import os
import argparse
import torch
from typing import Dict, List, Any

# 添加路径（项目根目录）- 必须在导入 router 之前
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)  # tests 目录
ROUTING_RAG_ROOT = os.path.dirname(CURRENT_DIR)  # 项目根目录

# 将项目根目录加入 sys.path
sys.path.insert(0, ROUTING_RAG_ROOT)

# 现在可以导入 router 模块了
from router.trainable_router.factory import TrainableRouterFactory
from router.trainable_router.config import TrainableRouterConfig


def load_router_model(model_path: str, config_path: str):
    """
    加载 Router 模型

    Args:
        model_path: 模型权重路径
        config_path: 配置文件路径（可选，从checkpoint优先）

    Returns:
        加载好的模型
    """
    print(f"加载 Router 模型: {model_path}")

    # 加载 checkpoint（包含完整的模型配置）
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # 从 checkpoint 读取配置
    checkpoint_config = checkpoint.get('config', {})
    print(f"Checkpoint config: {checkpoint_config}")

    # 构建配置对象
    model_config = {
        'model_type': checkpoint_config.get('model_type', 'dc'),
        'model': {
            'backbone_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'hidden_size': checkpoint.get('hidden_size', 384),
            'strategy_names': checkpoint.get('strategy_names', ['no_rag', 'naive_rag']),
            'num_strategies': len(checkpoint.get('strategy_names', ['no_rag', 'naive_rag'])),
            'similarity_function': checkpoint_config.get('similarity_function', 'cos'),
            'temperature': checkpoint_config.get('temperature', 1.0),
        },
        'training': {
            'batch_size': 32,
        },
        'data': {
            'source': 'hotpotqa',
        }
    }

    config = TrainableRouterConfig.from_dict(model_config)

    # 创建模型
    model = TrainableRouterFactory.create_model(config)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载完成: {config.model_type}")
    print(f"策略: {config.model.strategy_names}")

    return model


def evaluate_scores(no_rag_score: float, naive_rag_score: float) -> str:
    """
    根据分数评估最优策略
    
    Args:
        no_rag_score: no_rag的分数
        naive_rag_score: naive_rag的分数
    
    Returns:
        最优策略名称
    """
    if no_rag_score > naive_rag_score:
        return 'no_rag'
    elif naive_rag_score > no_rag_score:
        return 'naive_rag'
    else:
        # 分数相等，按照argmax规则返回第一个（no_rag）
        return 'no_rag'


def test_router_routing(model, test_file: str, no_rag_results: str = None,
                 naive_rag_results: str = None, num_samples: int = None):
    """
    测试 Router 的路由决策
    
    Args:
        model: Router 模型
        test_file: 测试数据文件路径（JSONL格式）
        no_rag_results: no_rag结果文件路径（可选，用于计算准确率）
        naive_rag_results: naive_rag结果文件路径（可选，用于计算准确率）
        num_samples: 测试样本数（None 表示全部）
        
    Returns:
        路由测试结果
    """
    print(f"\n读取测试数据: {test_file}")
    
    # 读取测试问题
    questions = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            
            try:
                data = json.loads(line)
                questions.append(data)
            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {i+1} 行失败: {e}")
                continue
    
    print(f"加载了 {len(questions)} 个问题")
    
    # 如果提供了策略结果，读取并计算准确率
    ground_truth = {}
    if no_rag_results and naive_rag_results:
        print(f"\n加载策略结果文件...")
        
        # 读取no_rag结果
        with open(no_rag_results, 'r', encoding='utf-8') as f:
            no_rag_data = json.load(f)
            for item in no_rag_data:
                qid = item.get('_id')
                if qid:
                    ground_truth[qid] = {
                        'no_rag': item,
                        'naive_rag': None
                    }
        
        # 读取naive_rag结果
        with open(naive_rag_results, 'r', encoding='utf-8') as f:
            naive_rag_data = json.load(f)
            for item in naive_rag_data:
                qid = item.get('_id')
                if qid in ground_truth:
                    ground_truth[qid]['naive_rag'] = item
        
        # 计算真实最优策略
        for qid, data in ground_truth.items():
            no_rag_item = data['no_rag']
            naive_rag_item = data['naive_rag']
            
            # 计算综合分数: 0.5 * EM + 0.5 * F1
            no_rag_score = 0.5 * no_rag_item.get('em', 0) + 0.5 * no_rag_item.get('f1', 0)
            naive_rag_score = 0.5 * naive_rag_item.get('em', 0) + 0.5 * naive_rag_item.get('f1', 0)
            
            ground_truth[qid]['true_strategy'] = evaluate_scores(no_rag_score, naive_rag_score)
            ground_truth[qid]['no_rag_score'] = no_rag_score
            ground_truth[qid]['naive_rag_score'] = naive_rag_score
        
        print(f"加载了 {len(ground_truth)} 个策略结果")
    
    # 路由决策统计
    route_stats = {
        'no_rag': 0,
        'naive_rag': 0
    }
    
    # 结果列表
    results = []
    correct_count = 0
    eval_count = 0
    
    # 路由决策统计
    route_stats = {
        'no_rag': 0,
        'naive_rag': 0
    }
    
    # 结果列表
    results = []
    
    # 进行路由预测
    for i, question_data in enumerate(questions):
        question = question_data['question']
        qid = question_data.get('_id', f'q{i}')
        
        # 编码问题
        if hasattr(model, 'encode'):
            query_emb = model.encode([question])
        else:
            print("错误: 模型不支持编码方法")
            return None
        
        # 获取策略 embedding
        strategy_emb = model.get_strategy_embeddings()
        
        # 计算相似度
        similarity = model.compute_similarity(query_emb, strategy_emb)
        
        # 选择最佳策略
        selected_strategy_idx = similarity.argmax(dim=-1).item()
        selected_strategy = model.strategy_names[selected_strategy_idx]
        confidence = similarity[0][selected_strategy_idx].item()
        
        # 更新统计
        route_stats[selected_strategy] += 1
        
        # 保存结果
        result = {
            '_id': qid,
            'question': question,
            'predicted_strategy': selected_strategy,
            'strategy_index': selected_strategy_idx,
            'confidence': float(confidence)
        }
        
        # 如果有真实标签，添加到结果中
        if qid in ground_truth:
            result['true_strategy'] = ground_truth[qid]['true_strategy']
            result['no_rag_score'] = ground_truth[qid]['no_rag_score']
            result['naive_rag_score'] = ground_truth[qid]['naive_rag_score']
            
            # 判断预测是否正确
            if selected_strategy == ground_truth[qid]['true_strategy']:
                correct_count += 1
                result['correct'] = True
            else:
                result['correct'] = False
            eval_count += 1
        
        results.append(result)
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(questions)} 个问题")
    
    # 计算统计
    total = len(results)
    no_rag_count = route_stats['no_rag']
    naive_rag_count = route_stats['naive_rag']
    
    statistics = {
        'total_questions': total,
        'no_rag_count': no_rag_count,
        'naive_rag_count': naive_rag_count,
        'no_rag_ratio': no_rag_count / total if total > 0 else 0.0,
        'naive_rag_ratio': naive_rag_count / total if total > 0 else 0.0
    }
    
    # 如果有真实标签，计算准确率
    if eval_count > 0:
        accuracy = correct_count / eval_count
        statistics['eval_count'] = eval_count
        statistics['correct_count'] = correct_count
        statistics['accuracy'] = accuracy
        
        # 按策略统计准确率
        no_rag_correct = sum(1 for r in results if r.get('correct', False) and r.get('true_strategy') == 'no_rag')
        naive_rag_correct = sum(1 for r in results if r.get('correct', False) and r.get('true_strategy') == 'naive_rag')
        
        no_rag_total = sum(1 for r in results if r.get('true_strategy') == 'no_rag')
        naive_rag_total = sum(1 for r in results if r.get('true_strategy') == 'naive_rag')
        
        statistics['no_rag_accuracy'] = no_rag_correct / no_rag_total if no_rag_total > 0 else 0.0
        statistics['naive_rag_accuracy'] = naive_rag_correct / naive_rag_total if naive_rag_total > 0 else 0.0
        
        # 策略召回率和精确率
        no_rag_recall = no_rag_correct / no_rag_total if no_rag_total > 0 else 0.0
        no_rag_precision = no_rag_correct / no_rag_count if no_rag_count > 0 else 0.0
        naive_rag_recall = naive_rag_correct / naive_rag_total if naive_rag_total > 0 else 0.0
        naive_rag_precision = naive_rag_correct / naive_rag_count if naive_rag_count > 0 else 0.0
        
        statistics['no_rag_recall'] = no_rag_recall
        statistics['no_rag_precision'] = no_rag_precision
        statistics['naive_rag_recall'] = naive_rag_recall
        statistics['naive_rag_precision'] = naive_rag_precision
        
        # F1分数
        statistics['no_rag_f1'] = 2 * no_rag_precision * no_rag_recall / (no_rag_precision + no_rag_recall) if (no_rag_precision + no_rag_recall) > 0 else 0.0
        statistics['naive_rag_f1'] = 2 * naive_rag_precision * naive_rag_recall / (naive_rag_precision + naive_rag_recall) if (naive_rag_precision + naive_rag_recall) > 0 else 0.0
    
    # 输出结果
    output = {
        'questions': results,
        'statistics': statistics
    }
    
    print(f"\n路由决策统计:")
    print(f"  no_rag: {no_rag_count} ({statistics['no_rag_ratio']:.2%})")
    print(f"  naive_rag: {naive_rag_count} ({statistics['naive_rag_ratio']:.2%})")
    
    if eval_count > 0:
        print(f"\n准确率评估:")
        print(f"  总准确率: {accuracy:.4f} ({correct_count}/{eval_count})")
        print(f"  no_rag准确率: {statistics.get('no_rag_accuracy', 0):.4f}")
        print(f"  naive_rag准确率: {statistics.get('naive_rag_accuracy', 0):.4f}")
        print(f"  no_rag F1: {statistics.get('no_rag_f1', 0):.4f}")
        print(f"  naive_rag F1: {statistics.get('naive_rag_f1', 0):.4f}")
    
    return output


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试 Router 模型的路由决策', 
                         formatter_class=argparse.RawDescriptionHelpFormatter,
                         epilog="""
使用示例:
  # 测试路由决策（不计算准确率）
  python test_router_routing.py --model router_models/no_rag_vs_naive/final/model.pt --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl
  
  # 测试路由决策并计算准确率
  python test_router_routing.py --model router_models/no_rag_vs_naive/final/model.pt --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl --no-rag-results HotpotQA/NoRag_results.json --naive-rag-results HotpotQA/Naiverag_results.json
  
  # 只测试前100个样本
  python test_router_routing.py --model router_models/no_rag_vs_naive/final/model.pt --test HotpotQA/hotpot_dev_distractor_1000_samples.jsonl --num-samples 100
                        """)
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重文件路径 (如: router_models/no_rag_vs_naive/final/model.pt)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (可选，默认从checkpoint读取)')
    parser.add_argument('--test', type=str, required=True,
                       help='测试数据文件路径 (JSONL格式)')
    parser.add_argument('--no-rag-results', type=str, default=None,
                       help='NoRAG结果文件路径 (JSON格式，用于计算准确率)')
    parser.add_argument('--naive-rag-results', type=str, default=None,
                       help='NaiveRAG结果文件路径 (JSON格式，用于计算准确率)')
    parser.add_argument('--output', type=str, default='tests/router_test_results.json',
                       help='输出文件路径 (默认: tests/router_test_results.json)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='测试样本数（None 表示全部）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Router 路由测试")
    print("="*60)
    print(f"模型路径: {args.model}")
    print(f"测试数据: {args.test}")
    if args.no_rag_results:
        print(f"NoRAG结果: {args.no_rag_results}")
    if args.naive_rag_results:
        print(f"NaiveRAG结果: {args.naive_rag_results}")
    print(f"输出文件: {args.output}")
    if args.num_samples:
        print(f"样本数: {args.num_samples}")
    print("="*60)
    
    # 加载模型
    model = load_router_model(args.model, args.config)
    
    # 测试路由
    output = test_router_routing(
        model, 
        args.test, 
        no_rag_results=args.no_rag_results,
        naive_rag_results=args.naive_rag_results,
        num_samples=args.num_samples
    )
    
    # 保存结果
    print(f"\n保存结果到: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("测试完成!")
    print("="*60)


if __name__ == '__main__':
    main()
