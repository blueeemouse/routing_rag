"""
调试GraphRAG的实际行为
检查context是否被正确检索和传递
"""
import sys
import os
from pathlib import Path

# 添加项目路径
ROUTING_RAG_ROOT = r"D:\Develop\all_RAG\routing_rag"
sys.path.insert(0, ROUTING_RAG_ROOT)

from rag_implementations.graph_rag.graph_rag_impl import GraphRAG

# 初始化GraphRAG
print("正在初始化GraphRAG...")
graphrag = GraphRAG()

# 准备context
data_path = r"D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data"
config_filename = "graphrag_hotpotqa_config.yml"

context = {
    'search_mode': 'local',
    'data_path': data_path,
    'config_filename': config_filename
}

# 测试查询 - 使用和评测结果相同的问题
test_queries = [
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
    "What science fantasy young adult series, told in first person, has a set of companion books narrating stories of enslaved worlds and alien species?",
    "Are Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
]

print("="*80)
print("GraphRAG调试测试")
print("="*80)
print(f"数据路径: {data_path}")
print(f"配置文件: {config_filename}")
print(f"测试问题数: {len(test_queries)}")
print("="*80)

for idx, query in enumerate(test_queries, 1):
    print(f"\n{'#'*80}")
    print(f"# 测试 {idx}/{len(test_queries)}")
    print(f"{'#'*80}")

    try:
        result = graphrag.execute(query, context=context)

        # 检查是否像"I don't know"这种没有context的回答
        result_lower = result.lower()
        if "i do not have" in result_lower or "i don't have" in result_lower or "i don't know" in result_lower:
            print(f"\n⚠️  警告：答案可能基于LLM先验知识而非检索context")
        elif "i'm sorry" in result_lower or "i'm sorry" in result_lower:
            print(f"\n⚠️  警告：答案可能基于LLM先验知识而非检索context")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)
