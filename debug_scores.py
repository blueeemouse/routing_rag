import json

norag = json.load(open('D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/Norag_results_20260127_012305.json', encoding='utf-8'))
naive = json.load(open('D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/Naiverag_results_20260127_012305.json', encoding='utf-8'))

print('NoRAG预测数量:', len(norag['results']['predictions']))
print('NaiveRAG预测数量:', len(naive['results']['predictions']))

print('\n前10个样本对比:')
no_rag_preds = norag['results']['predictions']
naive_rag_preds = naive['results']['predictions']

for i in range(10):
    no_rag_score = 0.5 * no_rag_preds[i]['em'] + 0.5 * no_rag_preds[i]['f1']
    naive_rag_score = 0.5 * naive_rag_preds[i]['em'] + 0.5 * naive_rag_preds[i]['f1']
    diff = abs(no_rag_score - naive_rag_score)
    print(f'{i}: no_rag={no_rag_score:.3f}, naive_rag={naive_rag_score:.3f}, diff={diff:.3f}')

print('\n差异分布统计:')
differences = []
for i in range(min(len(no_rag_preds), len(naive_rag_preds))):
    no_rag_score = 0.5 * no_rag_preds[i]['em'] + 0.5 * no_rag_preds[i]['f1']
    naive_rag_score = 0.5 * naive_rag_preds[i]['em'] + 0.5 * naive_rag_preds[i]['f1']
    differences.append(abs(no_rag_score - naive_rag_score))

import numpy as np
print(f'  平均差异: {np.mean(differences):.4f}')
print(f'  最小差异: {min(differences):.4f}')
print(f'  最大差异: {max(differences):.4f}')
print(f'  差异=0的数量: {sum(1 for d in differences if d < 0.001)}')
