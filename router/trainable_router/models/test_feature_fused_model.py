"""
测试FeatureFusedRouterModel
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

import torch
from router.trainable_router.config import ModelConfig
from router.trainable_router.models.feature_fused_model import FeatureFusedRouterModel


def test_feature_fused_model():
    """测试特征融合模型"""
    print("=" * 80)
    print("测试 FeatureFusedRouterModel")
    print("=" * 80)
    
    # 创建配置
    config = ModelConfig(
        backbone_name="sentence-transformers/all-MiniLM-L6-v2",  # 使用较小的模型测试
        hidden_size=384,
        strategy_names=["no_rag", "naive_rag"],
        num_strategies=2,
        temperature=0.5,
        device="cpu"
    )
    
    # 创建模型
    print("\n1. 创建模型...")
    model = FeatureFusedRouterModel(
        config,
        use_spacy=False,  # 测试时使用规则方法
        feature_normalize=True,
        use_projection=True
    )
    
    # 测试数据
    test_queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
    ]
    
    print("\n2. 测试前向传播...")
    # 模拟分词
    inputs = model.tokenizer(
        test_queries,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # 前向传播
    logits = model(
        inputs['input_ids'],
        inputs['attention_mask'],
        test_queries
    )
    
    print(f"  Input queries: {len(test_queries)}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits:\n{logits}")
    print(f"  Handcrafted feature dimension: {model.handcrafted_dim}")
    
    # 测试预测
    print("\n3. 测试预测...")
    routes = model.route(test_queries)
    print(f"  Predicted routes: {routes}")
    
    # 测试保存和加载
    print("\n4. 测试保存和加载...")
    test_save_path = "router_models/test_feature_fused"
    model.save(test_save_path)
    print(f"  模型已保存到: {test_save_path}")
    
    # 加载模型
    model2 = FeatureFusedRouterModel(
        config,
        use_spacy=False,
        feature_normalize=True,
        use_projection=True
    )
    model2.load(test_save_path)
    print(f"  模型已加载")
    
    # 验证加载后的预测
    routes2 = model2.route(test_queries)
    print(f"  加载后预测结果: {routes2}")
    
    print("\n" + "=" * 80)
    print("✓ 测试通过！")
    print("=" * 80)


if __name__ == '__main__':
    test_feature_fused_model()
