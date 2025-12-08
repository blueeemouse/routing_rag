"""
测试NaiveRAG重构后的功能
Test the refactored NaiveRAG functionality
"""
import sys
import os

# 将项目根目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG

def test_naive_rag_refactor():
    """测试重构后的NaiveRAG功能"""
    print("开始测试重构后的NaiveRAG功能...")
    
    # 创建NaiveRAG实例
    naive_rag = NaiveRAG()
    
    # 准备测试数据
    test_documents = [
        "人工智能（AI）是计算机科学的分支，旨在模拟人类智能，让机器完成原本需要人类思考的任务。",
        "RAG系统（检索增强生成）是一种结合外部知识库和语言模型的技术。",
        "机器学习是AI的一个分支，让机器从数据中学习规律和模式。"
    ]
    
    print("1. 测试build_index方法...")
    success = naive_rag.build_index(test_documents)
    if success:
        print("   ✅ build_index方法执行成功")
    else:
        print("   ❌ build_index方法执行失败")
        return False
    
    print("2. 测试execute方法（使用预构建的索引）...")
    query = "什么是人工智能？"
    response = naive_rag.execute(query)
    print(f"   查询: {query}")
    print(f"   响应: {response}")
    
    if response and "错误" not in response:
        print("   ✅ execute方法执行成功")
    else:
        print("   ❌ execute方法执行失败")
        return False
    
    print("3. 测试向后兼容性（在上下文中提供文档）...")
    # 创建一个新的实例来测试向后兼容性
    naive_rag_compat = NaiveRAG()
    context_with_docs = {
        'documents': ["这是一个用于向后兼容性测试的文档。"]
    }
    
    query_compat = "这个文档说了什么？"
    response_compat = naive_rag_compat.execute(query_compat, context_with_docs)
    print(f"   查询: {query_compat}")
    print(f"   响应: {response_compat}")
    
    if response_compat and "错误" not in response_compat:
        print("   ✅ 向后兼容性测试成功")
    else:
        print("   ❌ 向后兼容性测试失败")
        return False
    
    print("\n✅ 所有测试通过！NaiveRAG重构成功。")
    return True

if __name__ == "__main__":
    test_naive_rag_refactor()