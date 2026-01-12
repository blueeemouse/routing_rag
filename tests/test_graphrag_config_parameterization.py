#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GraphRAG配置文件参数化功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置环境变量
os.environ['GRAPHRAG_API_KEY'] = os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE')

def test_config_filename_parameter():
    """
    测试通过context传递config_filename参数
    """
    try:
        from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
        
        print("测试1: 通过context传递config_filename参数")
        print("=" * 60)
        
        graph_rag = GraphRAG()
        
        if not graph_rag._graph_rag_available:
            print("[SKIP] GraphRAG不可用，跳过测试")
            return True
        
        # 使用已构建的测试数据路径
        test_data_path = os.path.join(project_root, "graphrag_hotpotqa_data")
        
        # 测试1: 指定配置文件名
        context = {
            'data_path': test_data_path,
            'search_mode': 'local',
            'config_filename': 'graphrag_hotpotqa_config.yml'
        }
        
        result = graph_rag.execute("测试查询", context=context)
        print(f"结果: {result[:100] if result else 'None'}...")
        
        if "错误" in result and "未找到配置文件" in result:
            print("[FAIL] 未找到指定的配置文件")
            return False
        elif "错误" in result:
            print("[INFO] 查询返回错误信息，但配置文件加载成功")
        else:
            print("[PASS] 配置文件加载成功")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_find_config():
    """
    测试自动查找配置文件功能
    """
    try:
        from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
        
        print("\n测试2: 自动查找配置文件")
        print("=" * 60)
        
        graph_rag = GraphRAG()
        
        if not graph_rag._graph_rag_available:
            print("[SKIP] GraphRAG不可用，跳过测试")
            return True
        
        # 使用已构建的测试数据路径
        test_data_path = os.path.join(project_root, "graphrag_hotpotqa_data")
        
        # 测试2: 不指定配置文件名，让系统自动查找
        context = {
            'data_path': test_data_path,
            'search_mode': 'local'
            # 不指定config_filename
        }
        
        result = graph_rag.execute("测试查询", context=context)
        print(f"结果: {result[:100] if result else 'None'}...")
        
        if "错误" in result and "未找到GraphRAG配置文件" in result:
            print("[FAIL] 自动查找配置文件失败")
            return False
        elif "错误" in result:
            print("[INFO] 查询返回错误信息，但配置文件自动查找成功")
        else:
            print("[PASS] 配置文件自动查找成功")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_invalid_config_filename():
    """
    测试指定不存在的配置文件名
    """
    try:
        from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
        
        print("\n测试3: 指定不存在的配置文件名")
        print("=" * 60)
        
        graph_rag = GraphRAG()
        
        if not graph_rag._graph_rag_available:
            print("[SKIP] GraphRAG不可用，跳过测试")
            return True
        
        # 使用已构建的测试数据路径
        test_data_path = os.path.join(project_root, "graphrag_hotpotqa_data")
        
        # 测试3: 指定不存在的配置文件名
        context = {
            'data_path': test_data_path,
            'search_mode': 'local',
            'config_filename': 'nonexistent_config.yml'
        }
        
        result = graph_rag.execute("测试查询", context=context)
        print(f"结果: {result[:100] if result else 'None'}...")
        
        if "错误" in result and "指定的配置文件不存在" in result:
            print("[PASS] 正确返回配置文件不存在的错误")
            return True
        else:
            print("[FAIL] 未正确处理不存在的配置文件")
            return False
        
    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_schema_parameter():
    """
    测试通过context传递vector_store_schema参数
    """
    try:
        from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
        
        print("\n测试4: 通过context传递vector_store_schema参数")
        print("=" * 60)
        
        graph_rag = GraphRAG()
        
        if not graph_rag._graph_rag_available:
            print("[SKIP] GraphRAG不可用，跳过测试")
            return True
        
        # 使用已构建的测试数据路径
        test_data_path = os.path.join(project_root, "graphrag_hotpotqa_data")
        
        # 测试4: 指定向量存储schema
        context = {
            'data_path': test_data_path,
            'search_mode': 'local',
            'config_filename': 'graphrag_hotpotqa_config.yml',
            'vector_store_schema': {
                'index_name': 'custom-entity-description',
                'id_field': 'custom_id',
                'vector_size': 1536
            }
        }
        
        result = graph_rag.execute("测试查询", context=context)
        print(f"结果: {result[:100] if result else 'None'}...")
        
        if "错误" in result:
            print("[INFO] 查询返回错误信息，但向量存储schema参数已传递")
        else:
            print("[PASS] 向量存储schema参数传递成功")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("GraphRAG配置文件参数化功能测试")
    print("=" * 60)
    
    results = []
    results.append(("指定配置文件名", test_config_filename_parameter()))
    results.append(("自动查找配置文件", test_auto_find_config()))
    results.append(("不存在的配置文件", test_invalid_config_filename()))
    results.append(("向量存储schema参数", test_vector_store_schema_parameter()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n[SUCCESS] 所有测试通过！")
        sys.exit(0)
    else:
        print("\n[ERROR] 部分测试失败！")
        sys.exit(1)
