#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GraphRAG的向量维度推断功能
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置环境变量
os.environ['GRAPHRAG_API_KEY'] = os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE')

from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
from graphrag.config.load_config import load_config

def test_vector_dimension_inference():
    """测试向量维度推断"""
    
    print("=" * 80)
    print("测试向量维度推断功能")
    print("=" * 80)
    
    # 创建GraphRAG实例
    graph_rag = GraphRAG()
    
    # 加载配置
    test_data_path = Path(project_root) / "graphrag_ollama_hotpotqa_15_test_data"
    config_path = test_data_path / "graphrag_hotpotqa_config.yml"
    
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        return False
    
    config = load_config(root_dir=test_data_path, config_filepath=config_path)
    
    # 测试_get_vector_store_schema方法
    print("\n测试 _get_vector_store_schema 方法:")
    print("-" * 80)
    
    schema_config = graph_rag._get_vector_store_schema(config)
    
    print(f"推断的向量维度: {schema_config.vector_size}")
    print(f"索引名称: {schema_config.index_name}")
    print(f"ID字段: {schema_config.id_field}")
    print(f"向量字段: {schema_config.vector_field}")
    
    # 测试自定义配置覆盖
    print("\n测试自定义配置覆盖:")
    print("-" * 80)
    
    custom_context = {
        'vector_store_schema': {
            'vector_size': 3072  # 测试覆盖
        }
    }
    
    schema_config_custom = graph_rag._get_vector_store_schema(config, context=custom_context)
    print(f"自定义向量维度: {schema_config_custom.vector_size}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_vector_dimension_inference()
    if not success:
        print("\n❌ 测试失败！")
        sys.exit(1)
