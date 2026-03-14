"""
准备Cost-Performance Router的训练数据

从评估结果中提取：
- question
- no_rag_f1, naive_rag_f1
- no_rag_retrieval_time, naive_rag_retrieval_time
"""

import json
import os

BASE_DIR = "D:/Develop/all_RAG/routing_rag"
NAIVERAG_FILE = os.path.join(BASE_DIR, "HotpotQA_train_data/Naiverag_results_20260127_012305.json")
NORAG_FILE = os.path.join(BASE_DIR, "HotpotQA_train_data/Norag_results_20260127_012305.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "HotpotQA_train_data/label_analysis/cp_router_labels.json")

def load_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']['predictions']

def prepare_training_data():
    print("加载NaiveRAG评估结果...")
    naive_results = load_results(NAIVERAG_FILE)
    
    print("加载NoRAG评估结果...")
    no_rag_results = load_results(NORAG_FILE)
    
    print(f"NaiveRAG: {len(naive_results)} 条")
    print(f"NoRAG: {len(no_rag_results)} 条")
    
    no_rag_dict = {r['question']: r for r in no_rag_results}
    
    samples = []
    tie_count = 0
    naive_better_count = 0
    no_rag_better_count = 0
    
    for naive_result in naive_results:
        question = naive_result['question']
        
        if question not in no_rag_dict:
            print(f"警告: 找不到NoRAG结果 for question: {question[:50]}...")
            continue
        
        no_rag_result = no_rag_dict[question]
        
        naive_f1 = naive_result.get('f1', 0.0)
        no_rag_f1 = no_rag_result.get('f1', 0.0)
        
        naive_retrieval_time = naive_result.get('retrieval_time', 0.0)
        no_rag_retrieval_time = 0.0
        
        if naive_f1 > no_rag_f1:
            optimal_strategy = 'naive_rag'
            naive_better_count += 1
        elif no_rag_f1 > naive_f1:
            optimal_strategy = 'no_rag'
            no_rag_better_count += 1
        else:
            if naive_retrieval_time > 0:
                optimal_strategy = 'no_rag'
            else:
                optimal_strategy = 'tie'
            tie_count += 1
        
        sample = {
            'question': question,
            'optimal_strategy': optimal_strategy,
            'no_rag_f1': no_rag_f1,
            'naive_rag_f1': naive_f1,
            'no_rag_retrieval_time': no_rag_retrieval_time,
            'naive_rag_retrieval_time': naive_retrieval_time,
        }
        samples.append(sample)
    
    print(f"\n=== 训练数据统计 ===")
    print(f"总样本数: {len(samples)}")
    print(f"naive_rag更优: {naive_better_count}")
    print(f"no_rag更优: {no_rag_better_count}")
    print(f"tie: {tie_count}")
    
    output_data = {'samples': samples}
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练数据已保存到: {OUTPUT_FILE}")
    
    print("\n=== 样本示例 ===")
    for i, sample in enumerate(samples[:3]):
        print(f"\n样本 {i+1}:")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Optimal: {sample['optimal_strategy']}")
        print(f"  no_rag_f1: {sample['no_rag_f1']:.2f}, naive_rag_f1: {sample['naive_rag_f1']:.2f}")
        print(f"  no_rag_time: {sample['no_rag_retrieval_time']:.2f}s, naive_rag_time: {sample['naive_rag_retrieval_time']:.2f}s")

if __name__ == '__main__':
    prepare_training_data()