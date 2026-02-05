"""
分析tie_queries的详细情况：都答对 vs 都答错
"""

import json
import os
import sys

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_tie_details(tie_file):
    """
    分析tie_queries中，两者都答对和都答错的情况

    Args:
        tie_file: tie_queries.json文件路径
    """
    with open(tie_file, 'r', encoding='utf-8') as f:
        tie_queries = json.load(f)

    both_correct = 0
    both_wrong = 0
    both_correct_examples = []
    both_wrong_examples = []

    for query in tie_queries:
        if query['no_rag_em'] == 1.0 and query['naive_rag_em'] == 1.0:
            both_correct += 1
            if len(both_correct_examples) < 5:
                both_correct_examples.append(query)
        else:
            both_wrong += 1
            if len(both_wrong_examples) < 5:
                both_wrong_examples.append(query)

    print(f"Tie查询分析：")
    print(f"{'='*80}")
    print(f"两者都答对（EM=1.0）: {both_correct} ({both_correct/len(tie_queries)*100:.2f}%)")
    print(f"两者都答错（EM=0.0）: {both_wrong} ({both_wrong/len(tie_queries)*100:.2f}%)")
    print(f"{'='*80}\n")

    print("两者都答对的例子：")
    print(f"{'-'*80}")
    for i, ex in enumerate(both_correct_examples, 1):
        print(f"{i}. {ex['question']}")
        print(f"   NoRAG: EM={ex['no_rag_em']}, F1={ex['no_rag_f1']}")
        print(f"   NaiveRAG: EM={ex['naive_rag_em']}, F1={ex['naive_rag_f1']}")
        print()

    print("\n两者都答错的例子：")
    print(f"{'-'*80}")
    for i, ex in enumerate(both_wrong_examples, 1):
        print(f"{i}. {ex['question']}")
        print(f"   NoRAG: EM={ex['no_rag_em']}, F1={ex['no_rag_f1']}")
        print(f"   NaiveRAG: EM={ex['naive_rag_em']}, F1={ex['naive_rag_f1']}")
        print()

if __name__ == "__main__":
    tie_file = r"D:\Develop\all_RAG\routing_rag\HotpotQA_train_data\label_analysis\tie_queries.json"
    analyze_tie_details(tie_file)
