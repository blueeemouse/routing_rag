"""
根据已有的策略评估结果，计算Router在测试集上的准确率
"""

import json
import os

def compute_router_accuracy_on_testset(
    no_rag_file: str,
    naive_rag_file: str,
    output_file: str
):
    """
    根据no_rag和naive_rag的评估结果，计算最优策略标签
    
    Args:
        no_rag_file: NoRAG评估结果JSON文件
        naive_rag_file: NaiveRAG评估结果JSON文件
        output_file: 输出文件路径
    """
    print("="*80)
    print("Router路由准确率评估（基于测试集）")
    print("="*80)
    
    # 读取NoRAG结果
    print(f"\n读取NoRAG结果: {no_rag_file}")
    with open(no_rag_file, 'r', encoding='utf-8') as f:
        no_rag_json = json.load(f)
    # 数据在 results.predictions 字段
    no_rag_data = no_rag_json['results']['predictions'] if 'results' in no_rag_json else no_rag_json

    print(f"  样本数: {len(no_rag_data)}")

    # 读取NaiveRAG结果
    print(f"读取NaiveRAG结果: {naive_rag_file}")
    with open(naive_rag_file, 'r', encoding='utf-8') as f:
        naive_rag_json = json.load(f)
    # 数据在 results.predictions 字段
    naive_rag_data = naive_rag_json['results']['predictions'] if 'results' in naive_rag_json else naive_rag_json

    print(f"  样本数: {len(naive_rag_data)}")
    
    # 创建ID映射
    no_rag_map = {item['question']: item for item in no_rag_data}
    naive_rag_map = {item['question']: item for item in naive_rag_data}
    
    # 统计
    no_rag_better = 0
    naive_rag_better = 0
    equal = 0
    total = 0
    
    results = []
    
    # 遍历所有样本
    for question in no_rag_map.keys():
        if question not in naive_rag_map:
            print(f"警告: 问题 {question} 在NaiveRAG中不存在，跳过")
            continue

        total += 1

        no_rag_item = no_rag_map[question]
        naive_rag_item = naive_rag_map[question]

        # 计算综合分数
        no_rag_score = 0.5 * no_rag_item.get('em', 0) + 0.5 * no_rag_item.get('f1', 0)
        naive_rag_score = 0.5 * naive_rag_item.get('em', 0) + 0.5 * naive_rag_item.get('f1', 0)

        # 确定最优策略
        if no_rag_score > naive_rag_score:
            optimal_strategy = 'no_rag'
            no_rag_better += 1
        elif naive_rag_score > no_rag_score:
            optimal_strategy = 'naive_rag'
            naive_rag_better += 1
        else:
            # 分数相等，按训练规则选择no_rag
            optimal_strategy = 'no_rag'
            equal += 1

        results.append({
            'question': question,
            'optimal_strategy': optimal_strategy,
            'no_rag_score': no_rag_score,
            'naive_rag_score': naive_rag_score,
            'no_rag_em': no_rag_item.get('em', 0),
            'no_rag_f1': no_rag_item.get('f1', 0),
            'naive_rag_em': naive_rag_item.get('em', 0),
            'naive_rag_f1': naive_rag_item.get('f1', 0),
            'score_diff': abs(no_rag_score - naive_rag_score)
        })
    
    # 计算标签分布
    no_rag_count = sum(1 for r in results if r['optimal_strategy'] == 'no_rag')
    naive_rag_count = sum(1 for r in results if r['optimal_strategy'] == 'naive_rag')
    
    print(f"\n{'='*80}")
    print("标签分布统计")
    print("="*80)
    print(f"  总样本数: {total}")
    print(f"  no_rag最优: {no_rag_count} ({no_rag_count/total:.2%})")
    print(f"  naive_rag最优: {naive_rag_count} ({naive_rag_count/total:.2%})")
    print(f"  分数相等: {equal} ({equal/total:.2%})")
    
    # 保存结果
    output = {
        'total_samples': total,
        'no_rag_better': no_rag_better,
        'naive_rag_better': naive_rag_better,
        'equal_scores': equal,
        'no_rag_ratio': no_rag_count / total if total > 0 else 0.0,
        'naive_rag_ratio': naive_rag_count / total if total > 0 else 0.0,
        'samples': results
    }
    
    print(f"\n保存结果到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("完成!")
    print("="*80)
    
    return output

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='根据测试集评估结果计算路由标签')
    parser.add_argument('--no-rag-file', type=str, required=False,
                       help='NoRAG评估结果文件（JSON格式），不指定则自动查找最新文件')
    parser.add_argument('--naive-rag-file', type=str, required=False,
                       help='NaiveRAG评估结果文件（JSON格式），不指定则自动查找最新文件')
    parser.add_argument('--output', type=str,
                       default='evaluation_results/router_test_labels.json',
                       help='输出文件路径')
    parser.add_argument('--eval-base', type=str,
                       default='evaluation_results/HotpotQA_test_data_evaluation',
                       help='评估结果目录（默认：evaluation_results/HotpotQA_test_data_evaluation）')

    args = parser.parse_args()

    # 如果用户指定了文件，直接使用；否则自动查找最新文件
    if args.no_rag_file and args.naive_rag_file:
        no_rag_latest = args.no_rag_file
        naive_rag_latest = args.naive_rag_file
        print(f"使用用户指定的结果文件:")
        print(f"  NoRAG: {os.path.basename(no_rag_latest)}")
        print(f"  NaiveRAG: {os.path.basename(naive_rag_latest)}")
    else:
        # 查找最新的结果文件
        eval_base = args.eval_base

        # 查找NoRag结果
        no_rag_files = []
        for fname in os.listdir(eval_base):
            if 'No' in fname and ('rag' in fname or 'RAG' in fname):
                no_rag_files.append(os.path.join(eval_base, fname))

        # 查找NaiveRAG结果
        naive_rag_files = []
        for fname in os.listdir(eval_base):
            if 'Naive' in fname or 'naive' in fname:
                naive_rag_files.append(os.path.join(eval_base, fname))

        # 选择最新的
        if not no_rag_files:
            print(f"错误: 未找到NoRAG结果文件")
            return
        if not naive_rag_files:
            print(f"错误: 未找到NaiveRAG结果文件")
            return

        import time
        no_rag_latest = max(no_rag_files, key=os.path.getmtime)
        naive_rag_latest = max(naive_rag_files, key=os.path.getmtime)

        print(f"自动查找最新的结果文件:")
        print(f"  NoRAG: {os.path.basename(no_rag_latest)}")
        print(f"  NaiveRAG: {os.path.basename(naive_rag_latest)}")
        print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(naive_rag_latest)))}")
    
    # 计算标签
    output = compute_router_accuracy_on_testset(
        no_rag_latest,
        naive_rag_latest,
        args.output
    )
    
    print("\n标签分布摘要:")
    print("="*80)
    print(f"基于 {output['total_samples']} 个测试样本:")
    print(f"  应路由到no_rag: {output['no_rag_ratio']:.2%}")
    print(f"  应路由到naive_rag: {output['naive_rag_ratio']:.2%}")
    print(f"  差距明显: {output['equal_scores']/output['total_samples']:.2%}")
    print("="*80)

if __name__ == '__main__':
    main()
