"""
使用示例：统一数据集和DataAdapter

展示如何使用新的统一数据加载和分数计算功能
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from router.trainable_router.data_utils import (
    DataAdapter, 
    TrainingItem, 
    ScoreComputer,
    StrategyMatcher
)
from router.trainable_router.datasets.hotpotqa_dataset import GenericRouterDataset, RouterDataLoader
from router.trainable_router.config import TrainableRouterConfig


def demo_strategy_matcher():
    """演示策略名称匹配"""
    print("=" * 60)
    print("1. 策略名称匹配示例")
    print("=" * 60)
    
    test_names = [
        "NoRAG_results.json",
        "Naiverag_results.json",
        "Graphrag_results.json",
        "no_rag_v2",
        "naive-rag-embedding",
        "graph_rag_kg",
    ]
    
    for name in test_names:
        matched = StrategyMatcher.match(name)
        print(f"  {name:30} -> {matched}")
    
    print()


def demo_score_computer():
    """演示分数计算器"""
    print("=" * 60)
    print("2. 分数计算器示例")
    print("=" * 60)
    
    metrics = {
        'em': 1.0,
        'f1': 0.9,
        'total_time': 5.0,
        'retrieval_time': 1.0,
        'generation_time': 4.0,
    }
    
    # 预设公式
    formulas = ["em", "f1", "em_f1_avg", "em_f1_weighted"]
    
    for formula in formulas:
        computer = ScoreComputer(formula)
        score = computer.compute(metrics)
        print(f"  {formula:20}: {score:.4f}")
    
    # 自定义公式
    print("\n  自定义公式示例:")
    custom_formulas = [
        "em * 0.7 + f1 * 0.3",
        "em - 0.01 * total_time",
        "(em + f1) / 2 - 0.005 * total_time",
    ]
    
    for formula in custom_formulas:
        computer = ScoreComputer(formula)
        score = computer.compute(metrics)
        print(f"  {formula:40}: {score:.4f}")
    
    print()


def demo_data_adapter():
    """演示数据适配器"""
    print("=" * 60)
    print("3. 数据适配器示例")
    print("=" * 60)
    
    adapter = DataAdapter("em_f1_avg")
    
    # 假设的数据路径
    eval_dir = "evaluation_results/HotpotQA_test_data_evaluation"
    
    if os.path.exists(eval_dir):
        # 加载目录
        items = adapter.from_directory(eval_dir)
        print(f"  从目录加载了 {len(items)} 条数据")
        
        # 聚合
        aggregated = adapter.aggregate(items)
        print(f"  聚合后: {len(aggregated)} 条唯一问题")
        
        # 过滤
        filtered = adapter.filter_by_coverage(aggregated, min_strategies=3)
        print(f"  过滤后（覆盖3个策略）: {len(filtered)} 条")
        
        # 统计
        coverage = adapter.filter_by_coverage(aggregated, min_strategies=3)
        print(f"\n  各策略覆盖统计:")
        stats = {}
        for item in coverage:
            for strat in item.strategy_scores:
                stats[strat] = stats.get(strat, 0) + 1
        for strat, count in sorted(stats.items()):
            print(f"    {strat}: {count}")
    else:
        print(f"  目录不存在: {eval_dir}")
        print("  使用示例数据演示...")
    
    print()


def demo_generic_dataset():
    """演示统一数据集"""
    print("=" * 60)
    print("4. 统一数据集示例")
    print("=" * 60)
    
    # 创建配置
    config = TrainableRouterConfig(
        model_type="dc",
        model={
            "backbone": "sentence-transformers/all-MiniLM-L6-v2",
            "strategy_names": ["no_rag", "naive_rag", "graph_rag"],
        },
        data={
            "train_path": "evaluation_results/HotpotQA_test_data_evaluation",
            "num_clusters": 5,
            "normalize_scores": True,
        }
    )
    
    # 创建数据集
    dataset = GenericRouterDataset(config, score_formula="em_f1_avg")
    
    # 尝试加载数据
    try:
        dataset.load_data(config.data.train_path)
        
        # 获取示例数据
        sample = dataset[0]
        print(f"  问题: {sample['question'][:50]}...")
        print(f"  分数: {sample['scores']}")
        print(f"  Cluster: {sample['cluster_id']}")
        print(f"  数据集大小: {len(dataset)}")
        
    except FileNotFoundError as e:
        print(f"  数据路径不存在: {e}")
    
    print()


def demo_router_data_loader():
    """演示简化数据加载器"""
    print("=" * 60)
    print("5. 简化数据加载器示例")
    print("=" * 60)
    
    eval_dir = "evaluation_results/HotpotQA_test_data_evaluation"
    
    if os.path.exists(eval_dir):
        # 创建加载器
        loader = RouterDataLoader(
            data_path=eval_dir,
            strategy_names=["no_rag", "naive_rag", "graph_rag"],
            score_formula="em_f1_avg",
            batch_size=32,
            shuffle=True
        )
        
        print(f"  数据总数: {len(loader.items)}")
        print(f"  批量数: {len(loader)}")
        
        # 获取第一批
        batch = next(iter(loader))
        print(f"  第一批问题数: {len(batch['questions'])}")
        print(f"  第一批分数样例: {batch['scores'][0] if batch['scores'] else 'N/A'}")
    else:
        print(f"  目录不存在: {eval_dir}")
    
    print()


def demo_end_to_end():
    """端到端示例"""
    print("=" * 60)
    print("6. 端到端流程示例")
    print("=" * 60)
    
    # 1. 创建适配器
    adapter = DataAdapter("em_f1_avg")
    
    # 2. 加载数据
    eval_dir = "evaluation_results/HotpotQA_test_data_evaluation"
    
    if os.path.exists(eval_dir):
        items = adapter.from_directory(eval_dir)
        aggregated = adapter.aggregate(items)
        filtered = adapter.filter_by_coverage(aggregated, min_strategies=3)
        normalized = adapter.normalize_scores(filtered)
        
        print(f"  ✓ 加载数据: {len(items)} -> 聚合: {len(aggregated)} -> 过滤: {len(filtered)}")
        
        # 3. 创建数据集
        dataset = GenericRouterDataset(
            config=None,  # 简化示例
            score_formula="em_f1_avg",
            data_adapter=adapter
        )
        dataset.data = normalized
        
        print(f"  ✓ 数据集大小: {len(dataset)}")
        
        # 4. 展示样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  ✓ 样本示例:")
            print(f"      问题: {sample['question'][:40]}...")
            print(f"      分数: {sample['scores']}")
            print(f"      Cluster: {sample['cluster_id']}")
    else:
        print(f"  目录不存在: {eval_dir}")
    
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  统一数据集和DataAdapter使用示例")
    print("=" * 60 + "\n")
    
    # 检查数据目录
    eval_dir = "evaluation_results/HotpotQA_test_data_evaluation"
    if not os.path.exists(eval_dir):
        print(f"注意: 数据目录不存在，将使用模拟数据演示")
        print(f"请确保目录存在: {os.path.abspath(eval_dir)}\n")
    
    # 运行所有示例
    demo_strategy_matcher()
    demo_score_computer()
    demo_data_adapter()
    demo_generic_dataset()
    demo_router_data_loader()
    demo_end_to_end()
    
    print("=" * 60)
    print("  示例完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()