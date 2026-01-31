"""检查GraphRAG索引的统计信息"""
import sys
sys.path.insert(0, r'D:\Develop\all_RAG\routing_rag')

try:
    import pandas as pd

    print("="*80)
    print("GraphRAG索引统计信息")
    print("="*80)

    # 检查第一个索引
    print("\n[1] graphrag_ollama_hotpotqa_1000_test_data")
    print("-"*80)
    try:
        entities_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\entities.parquet')
        relationships_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\relationships.parquet')
        communities_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\communities.parquet')
        reports_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\community_reports.parquet')

        print(f"  实体数量: {len(entities_df)}")
        print(f"  关系数量: {len(relationships_df)}")
        print(f"  社区数量: {len(communities_df)}")
        print(f"  社区报告数量: {len(reports_df)}")

        if 'degree' in entities_df.columns:
            print(f"  平均实体度数: {entities_df['degree'].mean():.2f}")
    except FileNotFoundError as e:
        print(f"  错误: 文件未找到 - {e}")

    # 检查第二个索引
    print("\n[2] graphrag_ollama_hotpotqa_test_data")
    print("-"*80)
    try:
        entities_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\entities.parquet')
        relationships_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\relationships.parquet')
        communities_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\communities.parquet')
        reports_df = pd.read_parquet(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\community_reports.parquet')

        print(f"  实体数量: {len(entities_df)}")
        print(f"  关系数量: {len(relationships_df)}")
        print(f"  社区数量: {len(communities_df)}")
        print(f"  社区报告数量: {len(reports_df)}")

        if 'degree' in entities_df.columns:
            print(f"  平均实体度数: {entities_df['degree'].mean():.2f}")
    except FileNotFoundError as e:
        print(f"  错误: 文件未找到 - {e}")

    print("\n" + "="*80)

except ImportError:
    print("错误: 需要安装 pandas")
    print("请运行: pip install pandas")
