#!/usr/bin/env python3
"""
KNN Router 测试脚本

测试内容：
1. 加载训练好的模型
2. 对新查询进行路由
3. 测试概率预测
4. 测试 KNN 信息功能
"""

import sys
sys.path.insert(0, '.')

from router.trainable_router.models.knn_model import KNNRouterModel
from router.trainable_router.config import ModelConfig

def main():
    # 加载训练好的模型
    print('加载模型...')
    
    # 创建一个默认配置，load 方法会覆盖这些值
    config = ModelConfig(
        backbone_name='BAAI/bge-base-en-v1.5',
        strategy_names=['no_rag', 'naive_rag'],
        num_strategies=2,
    )
    model = KNNRouterModel(config)
    model.load('router_models/knn_test/final')
    
    # 测试路由功能
    test_queries = [
        'What is the capital of France?',
        'Who was the first president of the United States?',
        'Which magazine was started first Arthur Magazine or First for Women?',
        'The Oberoi family is part of a hotel company that has a head office in what city?',
    ]
    
    print('\n' + '='*60)
    print('测试路由功能:')
    print('='*60)
    for query in test_queries:
        route = model.route([query])[0]
        print(f'  Q: {query[:50]}...')
        print(f'  -> 路由策略: {route}')
        print()
    
    # 测试概率预测
    print('='*60)
    print('测试概率预测:')
    print('='*60)
    probas = model.predict_proba(test_queries)
    for query, proba in zip(test_queries, probas):
        print(f'  Q: {query[:40]}...')
        print(f'  -> 概率分布: no_rag={proba[0]:.3f}, naive_rag={proba[1]:.3f}')
        print()
    
    # 测试 KNN 信息
    print('='*60)
    print('测试 KNN 信息:')
    print('='*60)
    info = model.get_knn_info(test_queries[0])
    print(f'  Query: {info["query"]}')
    print(f'  预测策略: {info["predicted_strategy"]}')
    print(f'  最近邻 ({info["k"]} 个):')
    for neighbor in info['neighbors'][:3]:
        print(f'    - {neighbor["strategy"]} (distance={neighbor["distance"]:.4f}): {neighbor["query"][:50]}...')
    
    print('\n测试完成!')

if __name__ == '__main__':
    main()
