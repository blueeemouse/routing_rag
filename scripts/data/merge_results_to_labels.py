import json
import os

def load_predictions_from_file(filepath):
    """从结果文件中提取 predictions 数据"""
    print(f"正在读取: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'results' in data and 'predictions' in data['results']:
        return data['results']['predictions']
    else:
        raise ValueError(f"无法从 {filepath} 中找到 predictions 数据")

def merge_results(no_rag_file, naive_rag_file, output_dir, output_name="all_labels.json"):
    """合并 no_rag 和 naive_rag 结果，生成标签文件
    
    Args:
        no_rag_file: no_rag 结果文件路径
        naive_rag_file: naive_rag 结果文件路径
        output_dir: 输出目录
        output_name: 输出文件名 (默认 all_labels.json)
    """
    
    # 加载数据
    no_rag_predictions = load_predictions_from_file(no_rag_file)
    naive_rag_predictions = load_predictions_from_file(naive_rag_file)
    
    # 确保数据长度一致
    if len(no_rag_predictions) != len(naive_rag_predictions):
        raise ValueError(f"两个文件的数据长度不一致: {len(no_rag_predictions)} vs {len(naive_rag_predictions)}")
    
    print(f"共处理 {len(no_rag_predictions)} 条数据")
    
    # 创建 all_labels 格式
    all_labels = []
    no_tie_samples = []
    
    for i in range(len(no_rag_predictions)):
        no_rag_item = no_rag_predictions[i]
        naive_rag_item = naive_rag_predictions[i]
        
        # 确保问题一致
        if no_rag_item['question'] != naive_rag_item['question']:
            print(f"警告: 第 {i} 条数据问题不匹配")
            print(f"  no_rag: {no_rag_item['question']}")
            print(f"  naive_rag: {naive_rag_item['question']}")
            continue
        
        # 提取分数
        no_rag_f1 = no_rag_item.get('f1', 0.0)
        no_rag_em = no_rag_item.get('em', 0.0)
        naive_rag_f1 = naive_rag_item.get('f1', 0.0)
        naive_rag_em = naive_rag_item.get('em', 0.0)
        
        # 计算 score (f1 和 em 的平均值)
        no_rag_score = (no_rag_f1 + no_rag_em) / 2.0
        naive_rag_score = (naive_rag_f1 + naive_rag_em) / 2.0
        
        # 确定 label
        if abs(no_rag_score - naive_rag_score) < 0.01:  # 相差小于0.01认为是 tie
            label = "tie"
        elif no_rag_score > naive_rag_score:
            label = "no_rag"
        else:
            label = "naive_rag"
        
        # 构建 all_labels 条目
        label_item = {
            "index": i,
            "question": no_rag_item['question'],
            "optimal_strategy": label,
            "no_rag_score": no_rag_score,
            "naive_rag_score": naive_rag_score,
            "no_rag_em": no_rag_em,
            "no_rag_f1": no_rag_f1,
            "naive_rag_em": naive_rag_em,
            "naive_rag_f1": naive_rag_f1
        }
        all_labels.append(label_item)
        
        # 如果不是 tie，添加到 no_tie_samples
        if label != "tie":
            no_tie_samples.append({
                "question": no_rag_item['question'],
                "optimal_strategy": label
            })
    
    # 统计信息
    no_rag_count = sum(1 for item in all_labels if item['optimal_strategy'] == 'no_rag')
    naive_rag_count = sum(1 for item in all_labels if item['optimal_strategy'] == 'naive_rag')
    tie_count = sum(1 for item in all_labels if item['optimal_strategy'] == 'tie')
    
    print(f"\n统计信息:")
    print(f"  no_rag: {no_rag_count}")
    print(f"  naive_rag: {naive_rag_count}")
    print(f"  tie: {tie_count}")
    print(f"  no_tie 总计: {len(no_tie_samples)}")
    
    # 保存主文件 (使用 samples 包装格式)
    all_labels_file = os.path.join(output_dir, output_name)
    all_labels_data = {"samples": all_labels}
    with open(all_labels_file, 'w', encoding='utf-8') as f:
        json.dump(all_labels_data, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {all_labels_file}")
    
    # 保存 no_tie 文件 (自动生成文件名)
    base_name = output_name.replace('.json', '')
    no_tie_file = os.path.join(output_dir, f"{base_name}_no_tie.json")
    no_tie_data = {"samples": no_tie_samples}
    with open(no_tie_file, 'w', encoding='utf-8') as f:
        json.dump(no_tie_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {no_tie_file}")
    
    return all_labels_file, no_tie_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description='合并 no_rag 和 naive_rag 结果')
    parser.add_argument('--no_rag_file', type=str, 
                        default="/home/lhz/code/routing_rag/HotpotQA_train_data/Norag_results_20260322_132705_vllm_qwen.json",
                        help='no_rag 结果文件路径')
    parser.add_argument('--naive_rag_file', type=str,
                        default="/home/lhz/code/routing_rag/HotpotQA_train_data/Naiverag_results_20260321_225008_vllm_qwen.json",
                        help='naive_rag 结果文件路径')
    parser.add_argument('--output_dir', type=str,
                        default="/home/lhz/code/routing_rag/HotpotQA_train_data/label_analysis",
                        help='输出目录')
    parser.add_argument('--output_name', type=str,
                        default="all_labels_vllm_qwen.json",
                        help='输出文件名 (不含路径)')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.no_rag_file):
        print(f"错误: 文件不存在 {args.no_rag_file}")
        return
    if not os.path.exists(args.naive_rag_file):
        print(f"错误: 文件不存在 {args.naive_rag_file}")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行合并
    try:
        merge_results(args.no_rag_file, args.naive_rag_file, args.output_dir, args.output_name)
        print("\n处理完成!")
    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
