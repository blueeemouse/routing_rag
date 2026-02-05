"""
转换 all_labels.json 格式为 RouterLabelDataset 可用格式
"""

import json
import os

def convert_all_labels_format(input_path, output_path):
    """
    转换 all_labels.json 格式
    
    从:
    [
        {
            "index": 0,
            "question": "...",
            "label": "no_rag",
            "no_rag_score": 1.0,
            "naive_rag_score": 0.0,
            ...
        }
    ]
    
    转换为:
    {
        "samples": [
            {
                "question": "...",
                "optimal_strategy": "no_rag"
            }
        ]
    }
    """
    
    print(f"读取: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_labels = json.load(f)
    
    print(f"样本数: {len(all_labels)}")
    
    # 转换格式
    samples = []
    for item in all_labels:
        # 跳过 tie 样本（因为在 compute_loss 中会强制处理）
        if item['label'] == 'tie':
            continue
            
        samples.append({
            'question': item['question'],
            'optimal_strategy': item['label']
        })
    
    print(f"转换后样本数 (排除tie): {len(samples)}")
    
    # 保存
    output_data = {
        'samples': samples
    }
    
    print(f"保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("转换完成!")

if __name__ == "__main__":
    input_file = r"D:\Develop\all_RAG\routing_rag\HotpotQA_train_data\label_analysis\all_labels.json"
    output_file = r"D:\Develop\all_RAG\routing_rag\HotpotQA_train_data\label_analysis\all_labels_converted.json"
    
    convert_all_labels_format(input_file, output_file)
