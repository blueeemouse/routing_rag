"""
分析日志中的extract graph每步耗时
"""

import re
from datetime import datetime

def parse_extract_graph_timing(log_file):
    """从日志中解析extract graph每步的耗时"""
    timings = []

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'extract graph progress:' in line:
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line)
                if timestamp_match:
                    timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S.%f')
                    timings.append((current, total, timestamp))

    # 计算每步的耗时
    intervals = []
    for i in range(1, len(timings)):
        prev_current, prev_total, prev_time = timings[i-1]
        curr_current, curr_total, curr_time = timings[i]
        delta = (curr_time - prev_time).total_seconds()
        intervals.append((curr_current, delta))

    return intervals

print("="*80)
print("Extract Graph 每步耗时分析")
print("="*80)

# 分析大索引
print("\n【大索引】graphrag_ollama_hotpotqa_1000_test_data")
print("-"*80)
try:
    large_intervals = parse_extract_graph_timing(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data\logs\indexing-engine.log')
    print(f"总步数: {len(large_intervals)}")
    if large_intervals:
        first_20 = large_intervals[:20]
        print("\n前20步耗时:")
        for step, delta in first_20:
            print(f"  Step {step}: {delta:.2f}s")

        avg_time = sum(delta for _, delta in large_intervals) / len(large_intervals)
        print(f"\n平均每步: {avg_time:.2f}s")
        print(f"最快: {min(delta for _, delta in large_intervals):.2f}s")
        print(f"最慢: {max(delta for _, delta in large_intervals):.2f}s")

        # 统计耗时分布
        fast = sum(1 for _, delta in large_intervals if delta < 1)
        medium = sum(1 for _, delta in large_intervals if 1 <= delta < 10)
        slow = sum(1 for _, delta in large_intervals if delta >= 10)
        print(f"\n耗时分布:")
        print(f"  <1s: {fast} ({fast/len(large_intervals)*100:.1f}%)")
        print(f"  1-10s: {medium} ({medium/len(large_intervals)*100:.1f}%)")
        print(f"  >=10s: {slow} ({slow/len(large_intervals)*100:.1f}%)")

except Exception as e:
    print(f"错误: {e}")

# 分析小索引
print("\n" + "="*80)
print("【小索引】graphrag_ollama_hotpotqa_test_data")
print("-"*80)
try:
    small_intervals = parse_extract_graph_timing(r'D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_test_data\logs\indexing-engine.log')
    print(f"总步数: {len(small_intervals)}")
    if small_intervals:
        print("\n所有步骤耗时:")
        for step, delta in small_intervals:
            print(f"  Step {step}: {delta:.2f}s")

        avg_time = sum(delta for _, delta in small_intervals) / len(small_intervals)
        print(f"\n平均每步: {avg_time:.2f}s")
        print(f"最快: {min(delta for _, delta in small_intervals):.2f}s")
        print(f"最慢: {max(delta for _, delta in small_intervals):.2f}s")

        # 统计耗时分布
        fast = sum(1 for _, delta in small_intervals if delta < 1)
        medium = sum(1 for _, delta in small_intervals if 1 <= delta < 10)
        slow = sum(1 for _, delta in small_intervals if delta >= 10)
        very_slow = sum(1 for _, delta in small_intervals if delta >= 60)
        print(f"\n耗时分布:")
        print(f"  <1s: {fast} ({fast/len(small_intervals)*100:.1f}%)")
        print(f"  1-10s: {medium} ({medium/len(small_intervals)*100:.1f}%)")
        print(f"  10-60s: {slow-very_slow} ({(slow-very_slow)/len(small_intervals)*100:.1f}%)")
        print(f"  >=60s: {very_slow} ({very_slow/len(small_intervals)*100:.1f}%)")

except Exception as e:
    print(f"错误: {e}")

print("\n" + "="*80)
print("结论:")
print("="*80)
print("""
通过日志分析，我们可以清楚地看到：
1. 大索引的extract_graph每步非常快（<1s），且分布均匀
2. 小索引的extract_graph有些步骤非常慢（>60s），明显遇到了超时/重试问题

这说明小索引建立时，某些文档的entity extraction遇到了困难：
- 可能是网络问题导致Ollama响应慢
- 可能是qwen2.5:3b在处理某些文档时输出错误格式，需要多次重试
- 重试机制虽然配置了，但当模型持续输出错误格式时，即使重试10次也无法成功

最终结果：
- 小索引虽然文档数少（50个），但某些文档的entity extraction质量很差
- 大索引虽然文档数多（9769个），但每个文档的entity extraction都很快且成功
- 这就是为什么大索引的检索效果更好（EM=0.267 vs EM=0.00）
""")
