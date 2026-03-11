import json
import os

def convert_tie_to_no_rag(input_file, output_file=None):
    """
    将 all_labels.json 中的 tie 标签转换为 no_rag

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（如果为None，则覆盖原文件）
    """
    if not os.path.exists(input_file):
        raise ValueError(f"输入文件不存在: {input_file}")

    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data.get('samples', [])
    print(f"共 {len(samples)} 条样本")

    # 统计信息
    tie_count = 0
    no_rag_count = 0
    naive_rag_count = 0

    # 转换 tie 为 no_rag
    for sample in samples:
        if sample.get('optimal_strategy') == 'tie':
            sample['optimal_strategy'] = 'no_rag'
            tie_count += 1
        elif sample.get('optimal_strategy') == 'no_rag':
            no_rag_count += 1
        elif sample.get('optimal_strategy') == 'naive_rag':
            naive_rag_count += 1

    print(f"转换完成:")
    print(f"  tie -> no_rag: {tie_count} 条")
    print(f"  no_rag: {no_rag_count} 条")
    print(f"  naive_rag: {naive_rag_count} 条")
    print(f"  总计: {len(samples)} 条")

    # 保存文件
    if output_file is None:
        output_file = input_file
        print(f"覆盖原文件: {output_file}")
    else:
        print(f"保存到新文件: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("保存成功!")
    return output_file

def main():
    # 输入文件
    input_file = "D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/10000/all_labels.json"

    # 输出文件（如果不指定，则覆盖原文件）
    # output_file = "D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/10000/all_labels_tie_as_no_rag.json"
    output_file = None  # 覆盖原文件

    try:
        convert_tie_to_no_rag(input_file, output_file)
    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
