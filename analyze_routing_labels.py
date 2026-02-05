"""
分析训练数据的标签分布
根据公式 score = 0.5 * EM + 0.5 * F1 计算每个策略的分数，
比较后确定应该routing到哪个策略
"""

import json
import os

def load_results(file_path):
    """加载评测结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']['predictions']

def analyze_routing_labels(no_rag_file, naive_rag_file, output_dir):
    """
    分析标签分布

    Args:
        no_rag_file: NoRAG结果文件路径
        naive_rag_file: NaiveRAG结果文件路径
        output_dir: 输出目录
    """
    # 加载结果
    print("加载数据...")
    no_rag_predictions = load_results(no_rag_file)
    naive_rag_predictions = load_results(naive_rag_file)

    print(f"NoRAG: {len(no_rag_predictions)} 条数据")
    print(f"NaiveRAG: {len(naive_rag_predictions)} 条数据")

    # 确保数据对齐
    assert len(no_rag_predictions) == len(naive_rag_predictions), \
        f"数据量不一致: NoRAG={len(no_rag_predictions)}, NaiveRAG={len(naive_rag_predictions)}"

    # 分析标签
    print("\n分析标签分布...")
    no_rag_count = 0
    naive_rag_count = 0
    tie_count = 0

    no_rag_queries = []
    naive_rag_queries = []
    tie_queries = []

    for i, (no_rag_pred, naive_rag_pred) in enumerate(zip(no_rag_predictions, naive_rag_predictions)):
        # 确保问题是同一个
        assert no_rag_pred['question'] == naive_rag_pred['question'], \
            f"第{i}个问题不匹配"

        # 计算分数: score = 0.5 * EM + 0.5 * F1
        no_rag_score = 0.5 * no_rag_pred['em'] + 0.5 * no_rag_pred['f1']
        naive_rag_score = 0.5 * naive_rag_pred['em'] + 0.5 * naive_rag_pred['f1']

        # 确定标签
        if naive_rag_score > no_rag_score:
            label = 'naive_rag'
            naive_rag_count += 1
            naive_rag_queries.append({
                'index': i,
                'question': no_rag_pred['question'],
                'no_rag_score': no_rag_score,
                'naive_rag_score': naive_rag_score,
                'no_rag_em': no_rag_pred['em'],
                'no_rag_f1': no_rag_pred['f1'],
                'naive_rag_em': naive_rag_pred['em'],
                'naive_rag_f1': naive_rag_pred['f1']
            })
        elif no_rag_score > naive_rag_score:
            label = 'no_rag'
            no_rag_count += 1
            no_rag_queries.append({
                'index': i,
                'question': no_rag_pred['question'],
                'no_rag_score': no_rag_score,
                'naive_rag_score': naive_rag_score,
                'no_rag_em': no_rag_pred['em'],
                'no_rag_f1': no_rag_pred['f1'],
                'naive_rag_em': naive_rag_pred['em'],
                'naive_rag_f1': naive_rag_pred['f1']
            })
        else:
            # 分数相等
            label = 'tie'
            tie_count += 1
            tie_queries.append({
                'index': i,
                'question': no_rag_pred['question'],
                'no_rag_score': no_rag_score,
                'naive_rag_score': naive_rag_score,
                'no_rag_em': no_rag_pred['em'],
                'no_rag_f1': no_rag_pred['f1'],
                'naive_rag_em': naive_rag_pred['em'],
                'naive_rag_f1': naive_rag_pred['f1']
            })

    # 输出统计结果
    total = len(no_rag_predictions)
    print(f"\n{'='*80}")
    print("标签分布统计")
    print(f"{'='*80}")
    print(f"总数据量: {total}")
    print(f"Routing到NoRAG: {no_rag_count} ({no_rag_count/total*100:.2f}%)")
    print(f"Routing到NaiveRAG: {naive_rag_count} ({naive_rag_count/total*100:.2f}%)")
    print(f"分数相等: {tie_count} ({tie_count/total*100:.2f}%)")
    print(f"{'='*80}\n")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    print(f"保存结果到: {output_dir}")

    # 保存统计摘要
    summary = {
        'total_samples': total,
        'no_rag_count': no_rag_count,
        'naive_rag_count': naive_rag_count,
        'tie_count': tie_count,
        'no_rag_percentage': no_rag_count / total * 100,
        'naive_rag_percentage': naive_rag_count / total * 100,
        'tie_percentage': tie_count / total * 100,
        'score_formula': '0.5 * EM + 0.5 * F1'
    }

    summary_file = os.path.join(output_dir, 'label_distribution_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"统计摘要已保存到: {summary_file}")

    # 保存NoRAG查询
    if no_rag_queries:
        no_rag_file = os.path.join(output_dir, 'no_rag_queries.json')
        with open(no_rag_file, 'w', encoding='utf-8') as f:
            json.dump(no_rag_queries, f, ensure_ascii=False, indent=2)
        print(f"NoRAG查询已保存到: {no_rag_file}")

    # 保存NaiveRAG查询
    if naive_rag_queries:
        naive_rag_file = os.path.join(output_dir, 'naive_rag_queries.json')
        with open(naive_rag_file, 'w', encoding='utf-8') as f:
            json.dump(naive_rag_queries, f, ensure_ascii=False, indent=2)
        print(f"NaiveRAG查询已保存到: {naive_rag_file}")

    # 保存分数相等的查询
    if tie_queries:
        tie_file = os.path.join(output_dir, 'tie_queries.json')
        with open(tie_file, 'w', encoding='utf-8') as f:
            json.dump(tie_queries, f, ensure_ascii=False, indent=2)
        print(f"分数相等查询已保存到: {tie_file}")

    # 保存完整的标签分配
    labels = []
    for i, (no_rag_pred, naive_rag_pred) in enumerate(zip(no_rag_predictions, naive_rag_predictions)):
        no_rag_score = 0.5 * no_rag_pred['em'] + 0.5 * no_rag_pred['f1']
        naive_rag_score = 0.5 * naive_rag_pred['em'] + 0.5 * naive_rag_pred['f1']

        if naive_rag_score > no_rag_score:
            label = 'naive_rag'
        elif no_rag_score > naive_rag_score:
            label = 'no_rag'
        else:
            label = 'tie'

        labels.append({
            'index': i,
            'question': no_rag_pred['question'],
            'label': label,
            'no_rag_score': no_rag_score,
            'naive_rag_score': naive_rag_score,
            'no_rag_em': no_rag_pred['em'],
            'no_rag_f1': no_rag_pred['f1'],
            'naive_rag_em': naive_rag_pred['em'],
            'naive_rag_f1': naive_rag_pred['f1']
        })

    labels_file = os.path.join(output_dir, 'all_labels.json')
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"完整标签已保存到: {labels_file}")

    return summary

if __name__ == "__main__":
    # 文件路径
    train_data_dir = r"D:\Develop\all_RAG\routing_rag\HotpotQA_train_data"
    no_rag_file = os.path.join(train_data_dir, "Norag_results_20260127_012305.json")
    naive_rag_file = os.path.join(train_data_dir, "Naiverag_results_20260127_012305.json")

    # 输出目录
    output_dir = os.path.join(train_data_dir, "label_analysis")

    # 运行分析
    analyze_routing_labels(no_rag_file, naive_rag_file, output_dir)
