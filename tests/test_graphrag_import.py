#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GraphRAG类实例化的脚本
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_graphrag_import():
    try:
        from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
        print("GraphRAG类导入成功")

        # 创建实例以测试导入
        graph_rag = GraphRAG()
        print(f"GraphRAG可用性: {graph_rag._graph_rag_available}")

        if graph_rag._graph_rag_available:
            print("GraphRAG库导入成功")
            print("支持的搜索模式:")
            print("   - Local Search (本地搜索)")
            print("   - Global Search (全局搜索)")
            print("   - DRIFT Search (迭代推理搜索)")
            print("   - Basic Search (基础搜索)")
            return True
        else:
            print("GraphRAG库导入失败")
            return False
    except ImportError as e:
        print(f"导入错误: {e}")
        return False
    except Exception as e:
        print(f"其他错误: {e}")
        return False

if __name__ == "__main__":
    success = test_graphrag_import()
    if success:
        print("\nGraphRAG导入测试成功！")
    else:
        print("\nGraphRAG导入测试失败！")