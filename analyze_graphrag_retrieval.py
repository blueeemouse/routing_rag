"""分析GraphRAG的检索质量"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# 切换到脚本所在目录
script_dir = Path(__file__).parent
os.chdir(script_dir)

# 这里需要导入GraphRAG的配置和执行逻辑
# 先做一个简化的分析框架

def analyze_retrieval_quality(questions: List[Dict], name: str, max_samples: int = 10):
    """分析检索质量

    Args:
        questions: 问题列表
        name: 数据集名称
        max_samples: 分析的最大样本数
    """
    print(f"\n{'='*60}")
    print(f"{name} 检索质量分析（前{max_samples}条）")
    print(f"{'='*60}")

    print(f"\n分析维度:")
    print("1. 检索到的实体数量")
    print("2. 检索到的关系数量")
    print("3. 检索结果的token数量（上下文长度）")
    print("4. 检索耗时")
    print("5. 检索结果的相关性（手动评估）")

    # 展示前5个问题
    print(f"\n前5个问题:")
    for i, item in enumerate(questions[:5], 1):
        print(f"\n{i}. {item['question']}")
        print(f"   答案: {item.get('answer', 'N/A')}")
        if 'context' in item:
            context = item.get('context', [])
            print(f"   上下文文档数: {len(context) if isinstance(context, list) else 'N/A'}")

def compare_retrieval_results():
    """对比两种场景的检索结果"""
    print(f"\n{'='*60}")
    print(f"对比分析计划")
    print(f"{'='*60}")

    scenarios = [
        ("场景A: 15条数据索引 + 测试15条", "索引: hotpot_1000前15条", "测试: hotpot_1000前15条"),
        ("场景B: 1000条数据索引 + 测试1000条", "索引: hotpot_dev_distractor全部", "测试: hotpot_dev_distractor全部"),
        ("场景C: 1000条数据索引 + 测试15条", "索引: hotpot_dev_distractor全部", "测试: hotpot_1000前15条"),
    ]

    print(f"\n实验场景:")
    for i, (name, index, test) in enumerate(scenarios, 1):
        print(f"\n{i}. {name}")
        print(f"   {index}")
        print(f"   {test}")

    print(f"\n需要检查的指标:")
    print("1. 检索准确率 (Precision@K)")
    print("2. 检索召回率 (Recall@K)")
    print("3. 检索结果的多样性")
    print("4. 检索结果与答案的重叠度")
    print("5. 最终回答的EM/F1与检索质量的相关性")

# 加载测试数据
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 主程序
if __name__ == '__main__':
    print("GraphRAG检索质量分析工具")
    print("="*60)

    # 加载数据
    data_1000 = load_jsonl('HotpotQA/hotpot_1000_samples.jsonl')
    data_dev = load_jsonl('HotpotQA/hotpot_dev_distractor_1000_samples.jsonl')

    print(f"\n数据加载完成:")
    print(f"hotpot_1000: {len(data_1000)} 条")
    print(f"hotpot_dev_distractor: {len(data_dev)} 条")

    # 分析数据集特征
    analyze_retrieval_quality(data_1000, "hotpot_1000", max_samples=5)
    analyze_retrieval_quality(data_dev, "hotpot_dev_distractor", max_samples=5)

    # 对比计划
    compare_retrieval_results()

    print(f"\n{'='*60}")
    print(f"下一步行动建议:")
    print(f"{'='*60}")
    print("""
1. 运行 compare_datasets.py 检查数据集差异
2. 运行 check_graphrag_index.py 检查索引质量
3. 手动对比几种场景下的检索结果（打印前几个问题的检索内容）
4. 检查GraphRAG的配置参数（特别是 community_detection, embedding_model等）
5. 对比不同规模索引的性能差异（是否检索算法在大规模数据上有问题）

需要手动执行:
- 从GraphRAG的实际运行日志中提取检索结果
- 对比不同场景下的检索token数、实体数、关系数
    """)
