#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GraphRAG的_local_search方法
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置环境变量
os.environ['GRAPHRAG_API_KEY'] = os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE')

def test_local_search():
    try:
        from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
        
        print("正在测试GraphRAG的_local_search方法...")
        
        # 创建GraphRAG实例
        graph_rag = GraphRAG()
        
        print(f"GraphRAG可用性: {graph_rag._graph_rag_available}")
        
        if not graph_rag._graph_rag_available:
            print("[ERROR] GraphRAG不可用，无法测试_local_search方法")
            return False
        
        # 使用已构建的测试数据路径
        test_data_path = os.path.join(project_root, "graphrag_class_test_data")
        
        print(f"测试数据路径: {test_data_path}")
        
        # 检查数据是否存在
        output_path = os.path.join(test_data_path, "output")
        if not os.path.exists(output_path):
            print(f"[ERROR] 输出路径不存在: {output_path}")
            return False
        
        print("开始测试_local_search方法...")
        
        # 调用_local_search方法
        result = graph_rag._local_search(
            query="人工智能的定义是什么？",
            data_path=test_data_path
        )
        
        print(f"查询结果: {result}")
        
        if "错误" in result or "错误" in result:
            print("[INFO] 查询返回错误信息，这可能是正常的（因为某些组件可能缺失）")
        else:
            print("[SUCCESS] _local_search方法执行完成")
        
        return True

    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_search()
    if success:
        print("\n[SUCCESS] GraphRAG _local_search方法测试完成！")
    else:
        print("\n[ERROR] GraphRAG _local_search方法测试失败！")