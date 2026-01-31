"""使用pyarrow检查parquet文件统计信息"""
import os

try:
    import pyarrow.parquet as pq

    print("="*80)
    print("GraphRAG索引统计信息 (使用pyarrow)")
    print("="*80)

    # 检查第一个索引
    print("\n[1] graphrag_ollama_hotpotqa_1000_test_data")
    print("-"*80)
    entities_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\entities.parquet'
    relationships_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\relationships.parquet'
    communities_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\communities.parquet'
    reports_file = r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\community_reports.parquet'

    try:
        entities_table = pq.read_table(entities_file)
        relationships_table = pq.read_table(relationships_file)
        communities_table = pq.read_table(communities_file)
        reports_table = pq.read_table(reports_file)

        print(f"  实体数量: {len(entities_table)}")
        print(f"  关系数量: {len(relationships_table)}")
        print(f"  社区数量: {len(communities_table)}")
        print(f"  社区报告数量: {len(reports_table)}")

        # 检查实体列
        print(f"\n  实体数据列: {entities_table.column_names}")

        # 如果有degree列，计算平均度数
        if 'degree' in entities_table.column_names:
            degrees = entities_table.column('degree').to_pylist()
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            print(f"  平均实体度数: {avg_degree:.2f}")

        # 检查描述长度
        if 'description' in entities_table.column_names:
            descs = entities_table.column('description').to_pylist()
            avg_desc_len = sum(len(str(d)) for d in descs) / len(descs) if descs else 0
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
        entities_table = pq.read_table(entities_file)
        relationships_table = pq.read_table(relationships_file)
        communities_table = pq.read_table(communities_file)
        reports_table = pq.read_table(reports_file)

        print(f"  实体数量: {len(entities_table)}")
        print(f"  关系数量: {len(relationships_table)}")
        print(f"  社区数量: {len(communities_table)}")
        print(f"  社区报告数量: {len(reports_table)}")

        # 检查实体列
        print(f"\n  实体数据列: {entities_table.column_names}")

        # 如果有degree列，计算平均度数
        if 'degree' in entities_table.column_names:
            degrees = entities_table.column('degree').to_pylist()
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            print(f"  平均实体度数: {avg_degree:.2f}")

        # 检查描述长度
        if 'description' in entities_table.column_names:
            descs = entities_table.column('description').to_pylist()
            avg_desc_len = sum(len(str(d)) for d in descs) / len(descs) if descs else 0
            print(f"  平均实体描述长度: {avg_desc_len:.2f}")

    except FileNotFoundError as e:
        print(f"  错误: 文件未找到 - {e}")

    print("\n" + "="*80)

except ImportError:
    print("错误: 需要安装 pyarrow")
    print("请运行: pip install pyarrow")
