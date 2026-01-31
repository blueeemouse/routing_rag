"""检查parquet文件统计信息"""
import subprocess
import sys

# 尝试使用pip安装的pandas
result = subprocess.run([sys.executable, "-m", "pip", "show", "pandas"], capture_output=True, text=True)
if "Name: pandas" in result.stdout:
    # 找到pandas的安装路径
    for line in result.stdout.split('\n'):
        if line.startswith('Location:'):
            pandas_path = line.split(':', 1)[1].strip()
            sys.path.insert(0, pandas_path)
            break

try:
    import pandas as pd

    print("="*80)
    print("GraphRAG索引统计信息")
    print("="*80)

    # 检查第一个索引
    print("\n[1] graphrag_ollama_hotpotqa_1000_test_data")
    print("-"*80)
    entities_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\entities.parquet'
    relationships_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\relationships.parquet'
    communities_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\communities.parquet'
    reports_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\community_reports.parquet'

    try:
        entities_df = pd.read_parquet(entities_file)
        relationships_df = pd.read_parquet(relationships_file)
        communities_df = pd.read_parquet(communities_file)
        reports_df = pd.read_parquet(reports_file)

        print(f"  实体数量: {len(entities_df)}")
        print(f"  关系数量: {len(relationships_df)}")
        print(f"  社区数量: {len(communities_df)}")
        print(f"  社区报告数量: {len(reports_df)}")

        print(f"\n  实体数据列: {list(entities_df.columns)}")

        if 'degree' in entities_df.columns:
            avg_degree = entities_df['degree'].mean()
            print(f"  平均实体度数: {avg_degree:.2f}")

        if 'description' in entities_df.columns:
            avg_desc_len = entities_df['description'].astype(str).str.len().mean()
            print(f"  平均实体描述长度: {avg_desc_len:.2f}")

    except FileNotFoundError as e:
        print(f"  错误: 文件未找到 - {e}")

    # 检查第二个索引
    print("\n[2] graphrag_ollama_hotpotqa_test_data")
    print("-"*80)
    entities_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\entities.parquet'
    relationships_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\relationships.parquet'
    communities_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\communities.parquet'
    reports_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\community_reports.parquet'

    try:
        entities_df = pd.read_parquet(entities_file)
        relationships_df = pd.read_parquet(relationships_file)
        communities_df = pd.read_parquet(communities_file)
        reports_df = pd.read_parquet(reports_file)

        print(f"  实体数量: {len(entities_df)}")
        print(f"  关系数量: {len(relationships_df)}")
        print(f"  社区数量: {len(communities_df)}")
        print(f"  社区报告数量: {len(reports_df)}")

        print(f"\n  实体数据列: {list(entities_df.columns)}")

        if 'degree' in entities_df.columns:
            avg_degree = entities_df['degree'].mean()
            print(f"  平均实体度数: {avg_degree:.2f}")

        if 'description' in entities_df.columns:
            avg_desc_len = entities_df['description'].astype(str).str.len().mean()
            print(f"  平均实体描述长度: {avg_desc_len:.2f}")

    except FileNotFoundError as e:
        print(f"  错误: 文件未找到 - {e}")

    print("\n" + "="*80)

except ImportError as e:
    print(f"错误: 无法导入pandas - {e}")
