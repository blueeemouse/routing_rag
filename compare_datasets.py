"""对比两个HotpotQA数据集的特征差异"""
import json
from pathlib import Path
import os

# 切换到脚本所在目录
script_dir = Path(__file__).parent
os.chdir(script_dir)

# 读取两个数据集
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data_1000 = load_jsonl('HotpotQA/hotpot_1000_samples.jsonl')
data_dev = load_jsonl('HotpotQA/hotpot_dev_distractor_1000_samples.jsonl')

print(f"数据集1 (hotpot_1000): {len(data_1000)} 条")
print(f"数据集2 (hotpot_dev_distractor): {len(data_dev)} 条")

# 分析特征
def analyze_features(data, name):
    print(f"\n{'='*60}")
    print(f"{name} 特征分析")
    print(f"{'='*60}")

    # 问题长度分布
    q_lengths = [len(item['question'].split()) for item in data]
    print(f"\n问题长度（词数）:")
    print(f"  平均: {sum(q_lengths)/len(q_lengths):.2f}")
    print(f"  最小: {min(q_lengths)}")
    print(f"  最大: {max(q_lengths)}")
    print(f"  中位数: {sorted(q_lengths)[len(q_lengths)//2]}")

    # 答案长度分布
    a_lengths = [len(str(item['answer']).split()) if 'answer' in item else 0 for item in data]
    print(f"\n答案长度（词数）:")
    print(f"  平均: {sum(a_lengths)/len(a_lengths):.2f}")
    print(f"  最小: {min(a_lengths)}")
    print(f"  最大: {max(a_lengths)}")

    # 问题类型分布（根据是否包含某些关键词）
    multi_hop = sum(1 for item in data if any(word in item['question'].lower() for word in ['and', 'also', 'between', 'both', 'related']))
    print(f"\n可能的multi-hop问题: {multi_hop} ({multi_hop/len(data)*100:.2f}%)")

    # 支持文档数量
    if 'context' in data[0] or 'supporting_facts' in data[0]:
        context_lens = [len(item.get('context', [])) for item in data]
        print(f"\n支持文档数量:")
        print(f"  平均: {sum(context_lens)/len(context_lens):.2f}")
        print(f"  最小: {min(context_lens)}")
        print(f"  最大: {max(context_lens)}")

    # 展示前3个问题
    print(f"\n前3个问题示例:")
    for i, item in enumerate(data[:3], 1):
        print(f"\n{i}. {item['question'][:100]}{'...' if len(item['question']) > 100 else ''}")
        print(f"   答案: {item.get('answer', 'N/A')}")

analyze_features(data_1000, "数据集1 (hotpot_1000_samples)")
analyze_features(data_dev, "数据集2 (hotpot_dev_distractor_1000)")

# 对比两个数据集的差异
print(f"\n{'='*60}")
print(f"两个数据集的交集/并集分析")
print(f"{'='*60}")

questions_1000 = set(item['question'] for item in data_1000)
questions_dev = set(item['question'] for item in data_dev)

intersection = questions_1000 & questions_dev
print(f"\n交集问题数量: {len(intersection)}")
print(f"说明: 两个数据集 {'完全相同' if len(intersection) == len(data_1000) else '有重叠'}")
