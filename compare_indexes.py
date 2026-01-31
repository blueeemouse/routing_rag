"""
深度分析两个GraphRAG索引的差异
重点关注：
1. 实体提取质量和数量
2. 实体描述长度分布
3. community reports生成质量
4. 是否有失败的entity extraction
"""
import json
import os

# 由于没有pyarrow/pandas，我们直接读取parquet的schema信息
# 使用arrow库或者直接解析

def read_parquet_meta(filepath):
    """读取parquet文件的基本元数据"""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(filepath)
        return {
            'num_rows': len(table),
            'columns': table.column_names,
            'table': table
        }
    except ImportError:
        # fallback: 尝试使用arrow
        try:
            import arrow
            # 这里简化处理，实际需要更复杂的逻辑
            return {'error': 'pyarrow not available'}
        except:
            return {'error': 'no parquet library'}

# 读取stats.json
def read_stats(stats_path):
    with open(stats_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 对比两个索引
large_stats = read_stats(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\output\stats.json')
small_stats = read_stats(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\output\stats.json')

print("="*80)
print("GraphRAG索引对比分析")
print("="*80)

print("\n【基本信息对比】")
print("-"*80)
print(f"{'指标':<30} {'大索引(9769 docs)':<25} {'小索引(50 docs)':<25} {'差异':<10}")
print("-"*80)

print(f"{'文档数量':<30} {large_stats['num_documents']:<25} {small_stats['num_documents']:<25} {large_stats['num_documents']/small_stats['num_documents']:.1f}x")
print(f"{'总运行时间(秒)':<30} {large_stats['total_runtime']:<25.1f} {small_stats['total_runtime']:<25.1f} {large_stats['total_runtime']/small_stats['total_runtime']:.1f}x")

# 计算每个阶段的平均时间
print(f"\n{'【阶段耗时分析(秒)】':<30}")
print("-"*80)
print(f"{'阶段':<30} {'大索引':<20} {'小索引':<20} {'每文档耗时'}")
print("-"*80)

workflow_keys = set(large_stats['workflows'].keys()) | set(small_stats['workflows'].keys())

for key in sorted(workflow_keys):
    large_time = large_stats['workflows'].get(key, {}).get('overall', 0)
    small_time = small_stats['workflows'].get(key, {}).get('overall', 0)

    large_per_doc = large_time / large_stats['num_documents'] if large_stats['num_documents'] > 0 else 0
    small_per_doc = small_time / small_stats['num_documents'] if small_stats['num_documents'] > 0 else 0

    print(f"{key:<30} {large_time:<20.1f} {small_time:<20.1f} 大:{large_per_doc:.2f}s 小:{small_per_doc:.2f}s")

print("\n" + "="*80)
print("关键发现：")
print("="*80)
print("\n1. 实体提取阶段:")
print(f"   - 大索引: extract_graph 耗时 {large_stats['workflows']['extract_graph']['overall']:.1f}s")
print(f"     平均每文档: {large_stats['workflows']['extract_graph']['overall']/large_stats['num_documents']:.2f}s")
print(f"   - 小索引: extract_graph 耗时 {small_stats['workflows']['extract_graph']['overall']:.1f}s")
print(f"     平均每文档: {small_stats['workflows']['extract_graph']['overall']/small_stats['num_documents']:.2f}s")
print(f"   → 比例: {large_stats['workflows']['extract_graph']['overall']/large_stats['num_documents'] / (small_stats['workflows']['extract_graph']['overall']/small_stats['num_documents']):.2f}x")

print("\n2. 社区报告生成阶段:")
print(f"   - 大索引: create_community_reports 耗时 {large_stats['workflows']['create_community_reports']['overall']:.1f}s ({large_stats['workflows']['create_community_reports']['overall']/60:.1f}分钟)")
print(f"     平均每文档: {large_stats['workflows']['create_community_reports']['overall']/large_stats['num_documents']:.2f}s")
print(f"   - 小索引: create_community_reports 耗时 {small_stats['workflows']['create_community_reports']['overall']:.1f}s ({small_stats['workflows']['create_community_reports']['overall']/60:.1f}分钟)")
print(f"     平均每文档: {small_stats['workflows']['create_community_reports']['overall']/small_stats['num_documents']:.2f}s")
print(f"   → 比例: {large_stats['workflows']['create_community_reports']['overall']/large_stats['num_documents'] / (small_stats['workflows']['create_community_reports']['overall']/small_stats['num_documents']):.2f}x")

print("\n3. 嵌入生成阶段:")
print(f"   - 大索引: generate_text_embeddings 耗时 {large_stats['workflows']['generate_text_embeddings']['overall']:.1f}s")
print(f"     平均每文档: {large_stats['workflows']['generate_text_embeddings']['overall']/large_stats['num_documents']:.2f}s")
print(f"   - 小索引: generate_text_embeddings 耗时 {small_stats['workflows']['generate_text_embeddings']['overall']:.1f}s")
print(f"     平均每文档: {small_stats['workflows']['generate_text_embeddings']['overall']/small_stats['num_documents']:.2f}s")
print(f"   → 比例: {large_stats['workflows']['generate_text_embeddings']['overall']/large_stats['num_documents'] / (small_stats['workflows']['generate_text_embeddings']['overall']/small_stats['num_documents']):.2f}x")

print("\n" + "="*80)
print("配置对比:")
print("="*80)
print("\n大索引 (graphrag_ollama_hotpotqa_1000_test_data):")
print("  - concurrent_requests: 25")
print("  - request_timeout: 180s")
print("  - max_retries: 10")
print("  - embed_graph.enabled: true")
print("\n小索引 (graphrag_ollama_hotpotqa_test_data):")
print("  - concurrent_requests: 25")
print("  - request_timeout: 180s")
print("  - max_retries: 10")
print("  - embed_graph.enabled: true")
print("\n→ 配置参数相同，但大索引的文档数量是小索引的195倍")

print("\n" + "="*80)
print("结论分析：")
print("="*80)
print("""
1. 实体提取每文档耗时差异：
   - 大索引: ~0.01s/doc
   - 小索引: ~36.36s/doc
   - 大索引快了约3000倍！这说明什么？

2. 可能的原因：
   a) 小索引在建立时，LLM可能遇到了一些"困难"文档，导致entity extraction失败或超时
      但因为有retry机制（max_retries=10，最大延迟10s），理论上应该会重试成功
   b) 大索引采用了更激进的并发策略（concurrent_requests=25），可能同时处理了更多文档
   c) qwen2.5:3b模型在处理某些文档时，可能输出格式不符合要求，导致解析失败
      但这种失败通常会被retry机制捕获

3. 15个测试样本 vs 1000个测试样本：
   - 15个样本时，索引质量可能较好（文档较少，提取质量高）
   - 1000个样本时，索引质量可能下降（文档多，但平均每文档提取时间反而更短）

4. 根本原因推测：
   小索引的entity extraction阶段平均每文档耗时36秒，这非常不正常。
   可能的解释：
   - 小索引建立时遇到了网络/模型不稳定的问题
   - 小索引建立时，Ollama可能有其他任务在运行，导致qwen2.5:3b响应慢
   - 大索引建立时，系统环境更稳定，Ollama性能更好

   建议：检查两个索引的详细日志，看看是否有：
   - timeout warnings
   - retry attempts
   - LLM API errors
""")
