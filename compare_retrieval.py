"""直接对比GraphRAG在两个场景下的检索结果"""
import json
import os
from pathlib import Path

# 切换到脚本所在目录
script_dir = Path(__file__).parent
os.chdir(script_dir)

print("="*80)
print("GraphRAG 检索结果对比分析")
print("="*80)

# 读取两个测试数据
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data_1000 = load_jsonl('HotpotQA/hotpot_1000_samples.jsonl')
data_dev = load_jsonl('HotpotQA/hotpot_dev_distractor_1000_samples.jsonl')

print(f"\n数据加载完成:")
print(f"  hotpot_1000: {len(data_1000)} 条")
print(f"  hotpot_dev_distractor: {len(data_dev)} 条")

# 选择重叠的问题进行对比
questions_1000 = {item['question']: item for item in data_1000}
questions_dev = {item['question']: item for item in data_dev}

overlap = set(questions_1000.keys()) & set(questions_dev.keys())
print(f"\n重叠问题数: {len(overlap)}")

# 选择3个重叠问题进行手动对比
sample_questions = list(overlap)[:3]

print(f"\n对比问题示例:")
for i, q in enumerate(sample_questions, 1):
    print(f"\n{i}. {q}")
    print(f"   答案 (1000): {questions_1000[q].get('answer', 'N/A')}")
    print(f"   答案 (dev):   {questions_dev[q].get('answer', 'N/A')}")

print(f"\n{'='*80}")
print("分析计划:")
print("="*80)
print("""
现在需要手动运行GraphRAG，记录检索结果：

步骤1: 用 graphrag_ollama_hotpotqa_test_data (15条索引) 回答前3个重叠问题
       → 记录检索到的实体数量、关系数量、上下文token数

步骤2: 用 graphrag_ollama_hotpotqa_1000_test_data (1000条索引) 回答相同3个问题
       → 记录检索到的实体数量、关系数量、上下文token数

步骤3: 对比两种场景的检索结果

关注指标：
1. 检索到的实体名称是否相关？
2. 检索到的关系是否包含答案相关信息？
3. 上下文长度是否合理？
4. 是否检索到了噪声信息？

如何手动获取检索结果：
方法1: 在GraphRAG代码中添加日志，打印 build_context 的返回
方法2: 使用 verbose 模式运行，观察中间输出
方法3: 查看GraphRAG的日志文件
""")

print("\n关键配置差异回顾:")
print(f"{'='*80}")
print("""
15条索引 (graphrag_ollama_hotpotqa_test_data):
  - embed_graph.enabled: true
  - concurrent_requests: 25
  
1000条索引 (graphrag_ollama_hotpotqa_1000_test_data):
  - embed_graph.enabled: false  ← 关键差异
  - concurrent_requests: 16

embed_graph 的作用：
  - 对图谱进行node2vec嵌入，生成节点的向量表示
  - 用于全局搜索和某些检索策略
  - 但 local_search 使用的是 entity description embedding（来自text embedding）

所以 embed_graph.enabled 可能不是主要原因！

需要检查的其他配置：
1. 检索参数 (top_k, max_context_tokens等)
2. 社区检测参数
3. 实体提取参数
""")
