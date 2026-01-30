#!/usr/bin/env python3
"""
路由器评估脚本

评估训练好的路由器的性能
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.trainable_router.routers.dc_router import DCRouter
from router.trainable_router.datasets.hotpotqa_dataset import HotpotQARouterDataset
from router.trainable_router.config import TrainableRouterConfig


def evaluate_router(
    router: DCRouter,
    test_data: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    评估路由器
    
    Args:
        router: 路由器实例
        test_data: 测试数据列表
        
    Returns:
        评估指标
    """
    questions = [item['question'] for item in test_data]
    true_best_strategies = []
    
    for item in test_data:
        scores = item.get('scores', {})
        if scores:
            best_strategy = max(scores.keys(), key=lambda k: scores[k])
            true_best_strategies.append(best_strategy)
        else:
            true_best_strategies.append('no_rag')
    
    # 预测
    predicted_strategies = router.route_batch(questions)
    
    # 计算准确率
    correct = sum(p == t for p, t in zip(predicted_strategies, true_best_strategies))
    accuracy = correct / len(true_best_strategies) if true_best_strategies else 0.0
    
    # 计算每个策略的准确率
    strategy_correct = defaultdict(int)
    strategy_total = defaultdict(int)
    
    for pred, true in zip(predicted_strategies, true_best_strategies):
        strategy_total[true] += 1
        if pred == true:
            strategy_correct[true] += 1
    
    strategy_accuracy = {}
    for strategy in strategy_correct:
        strategy_accuracy[strategy] = strategy_correct[strategy] / strategy_total[strategy] if strategy_total[strategy] > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(true_best_strategies),
        'strategy_accuracy': dict(strategy_accuracy),
    }


def main():
    parser = argparse.ArgumentParser(description='评估路由器')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--test_data', type=str, required=True, help='测试数据路径或目录')
    parser.add_argument('--output', type=str, default='', help='输出结果路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"模型不存在: {args.model_path}")
        return
    
    # 创建路由器
    print(f"加载模型: {args.model_path}")
    router = DCRouter(args.model_path)
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_data}")
    
    # 使用HotpotQADataset加载数据
    config = TrainableRouterConfig(
        model_type='dc',
        model=TrainableRouterConfig().model,
        data=TrainableRouterConfig().data,
    )
    dataset = HotpotQARouterDataset(config)
    dataset.load_data(args.test_data)
    
    test_data = dataset.data
    print(f"测试样本数: {len(test_data)}")
    
    # 评估
    print("评估中...")
    metrics = evaluate_router(router, test_data)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"准确率: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print("\n各策略准确率:")
    for strategy, acc in metrics['strategy_accuracy'].items():
        print(f"  {strategy}: {acc:.4f}")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
