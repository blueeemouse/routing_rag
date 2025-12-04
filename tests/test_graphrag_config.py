#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GraphRAG配置加载功能的脚本
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_graphrag_config():
    try:
        from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
        from config.config import settings
        
        print("正在测试GraphRAG配置加载功能...")
        
        # 创建GraphRAG实例
        graph_rag = GraphRAG()
        
        # 检查配置参数是否正确加载
        print(f"API URL: {graph_rag.api_url}")
        print(f"API Key: {'*' * len(graph_rag.api_key) if graph_rag.api_key else 'Not Set'}")  # 隐藏实际API密钥
        print(f"Model: {graph_rag.model}")
        print(f"Embedding Model: {graph_rag.embedding_model}")
        print(f"Chunk Size: {graph_rag.chunk_size}")
        print(f"Top K: {graph_rag.top_k}")
        print(f"Temperature: {graph_rag.temperature}")
        
        # 验证是否从settings正确加载了配置
        assert graph_rag.api_url == settings.graph_rag_api_url, "API URL未正确加载"
        assert graph_rag.model == settings.graph_rag_model, "Model未正确加载"
        assert graph_rag.embedding_model == settings.graph_rag_embedding_model, "Embedding Model未正确加载"
        assert graph_rag.chunk_size == settings.graph_rag_chunk_size, "Chunk Size未正确加载"
        assert graph_rag.top_k == settings.graph_rag_top_k, "Top K未正确加载"
        assert graph_rag.temperature == settings.graph_rag_temperature, "Temperature未正确加载"
        
        print("\n[SUCCESS] 所有配置参数都已正确从settings加载！")
        return True

    except AssertionError as e:
        print(f"\n[ERROR] 配置加载验证失败: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = test_graphrag_config()
    if success:
        print("\n[SUCCESS] GraphRAG配置加载测试成功！")
    else:
        print("\n[ERROR] GraphRAG配置加载测试失败！")