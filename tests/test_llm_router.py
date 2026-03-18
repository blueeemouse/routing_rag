"""
LLM Router 测试脚本

测试 LLMRouter 的 zero-shot 和 few-shot 模式
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router.llm_router import LLMRouter
from config.config import settings


def test_zero_shot():
    """测试 zero-shot 模式"""
    print("=" * 50)
    print("测试 Zero-Shot 模式")
    print("=" * 50)
    
    # 从配置创建
    config = settings.llm_router_config
    config['mode'] = 'zero_shot'
    
    router = LLMRouter.from_config(config)
    
    # 测试几个样例
    test_queries = [
        "1+1等于几？",
        "北京是中国的首都吗？",
        "谁是2020年美国总统？",
        "爱因斯坦的相对论是什么？",
        "Python的创始人是谁？",
    ]
    
    for query in test_queries:
        result = router.route(query)
        print(f"查询: {query}")
        print(f"路由: {result}")
        print("-" * 30)
    
    return router


def test_few_shot():
    """测试 few-shot 模式"""
    print("\n" + "=" * 50)
    print("测试 Few-Shot 模式")
    print("=" * 50)
    
    # 从配置创建
    config = settings.llm_router_config
    config['mode'] = 'few_shot'
    config['few_shot_k'] = 3
    
    router = LLMRouter.from_config(config)
    
    # 计算样例总数
    total_examples = sum(len(v) for v in router.few_shot_examples_by_strategy.values())
    print(f"加载了 {total_examples} 个样例")
    for strategy, items in router.few_shot_examples_by_strategy.items():
        if items:
            print(f"  - {strategy}: {len(items)} 样例")
    
    # 测试几个样例
    test_queries = [
        "谁是2020年美国总统？",
        "Python的创始人是谁？",
        "Are both Cypress and Ajuga genera?",
    ]
    
    for query in test_queries:
        result = router.route(query)
        print(f"查询: {query}")
        print(f"路由: {result}")
        print("-" * 30)
    
    return router


def evaluate_on_test_data(router, test_file: str, sample_size: int = 50):
    """在测试数据上评估路由器"""
    print("\n" + "=" * 50)
    print(f"评估路由器性能 (采样 {sample_size} 条)")
    print("=" * 50)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples'][:sample_size]
    
    correct = 0
    total = 0
    
    # 只评估 no_rag 和 naive_rag 的样例
    for sample in samples:
        true_label = sample.get('optimal_strategy')
        if true_label not in ['no_rag', 'naive_rag']:
            continue
        
        query = sample['question']
        pred_label = router.route(query)
        
        total += 1
        if pred_label == true_label:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} 查询: {query[:50]}...")
        print(f"   真实: {true_label}, 预测: {pred_label}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n准确率: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy


def main():
    """主测试函数"""
    print("LLM Router 测试\n")
    
    # 检查配置
    print("当前配置:")
    print(f"  API URL: {settings.llm_router_api_url}")
    print(f"  Model: {settings.llm_router_model}")
    print(f"  Mode: {settings.llm_router_mode}")
    print()
    
    # 测试 zero-shot
    try:
        zero_shot_router = test_zero_shot()
    except Exception as e:
        print(f"Zero-shot 测试失败: {e}")
        zero_shot_router = None
    
    # 测试 few-shot
    try:
        few_shot_router = test_few_shot()
    except Exception as e:
        print(f"Few-shot 测试失败: {e}")
        few_shot_router = None
    
    # 评估性能
    test_file = "evaluation_results/router_test_labels.json"
    if os.path.exists(test_file):
        print("\n" + "=" * 50)
        print("性能评估")
        print("=" * 50)
        
        if zero_shot_router:
            print("\nZero-Shot 模式评估:")
            evaluate_on_test_data(zero_shot_router, test_file, sample_size=30)
        
        if few_shot_router:
            print("\nFew-Shot 模式评估:")
            evaluate_on_test_data(few_shot_router, test_file, sample_size=30)
    else:
        print(f"测试文件不存在: {test_file}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
