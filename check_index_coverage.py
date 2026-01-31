"""
检查GraphRAG索引是否包含测试集所需的文档
用于诊断性能差异问题
"""

import os
import json
import argparse

def extract_test_titles(hotpotqa_file: str, num_samples: int = None) -> set[str]:
    """
    从HotpotQA测试文件中提取所有文档标题

    Args:
        hotpotqa_file: HotpotQA数据文件路径（jsonl格式）
        num_samples: 限制处理的样本数量，None表示处理全部

    Returns:
        文档标题集合
    """
    titles = set()
    count = 0

    with open(hotpotqa_file, 'r', encoding='utf-8') as f:
        for line in f:
            if num_samples and count >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                context = data.get('context', [])

                # 提取每个文档标题
                for title, _ in context:
                    # 标准化标题（去除文件系统不允许的字符）
                    safe_title = title.replace('/', '_').replace('\\', '_').replace(':', '_')
                    for char in '<>:"\\|?*':
                        safe_title = safe_title.replace(char, '_')
                    titles.add(safe_title)

                count += 1
            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue

    return titles

def get_index_docs(input_dir: str) -> set[str]:
    """
    获取索引input目录中的所有文档标题（不含.txt后缀）

    Args:
        input_dir: GraphRAG的input目录路径

    Returns:
        文档标题集合（不含.txt后缀）
    """
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return set()

    docs = set()
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            # 去掉.txt后缀
            doc_title = filename[:-4]
            docs.add(doc_title)

    return docs

def check_coverage(test_titles: set[str], index_docs: set[str]) -> dict:
    """
    检查索引对测试集的覆盖情况

    Args:
        test_titles: 测试集所需的文档标题集合
        index_docs: 索引中包含的文档标题集合

    Returns:
        包含统计信息的字典
    """
    coverage = {
        'test_total': len(test_titles),
        'index_total': len(index_docs),
        'matched': len(test_titles & index_docs),
        'missing': len(test_titles - index_docs),
        'extra': len(index_docs - test_titles),
        'matched_titles': sorted(test_titles & index_docs),
        'missing_titles': sorted(test_titles - index_docs),
    }

    coverage['coverage_rate'] = coverage['matched'] / coverage['test_total'] if coverage['test_total'] > 0 else 0

    return coverage

def main():
    parser = argparse.ArgumentParser(description='检查GraphRAG索引是否包含测试集所需的文档')
    parser.add_argument('--test_file', type=str, required=True,
                        help='HotpotQA测试文件路径（jsonl格式）')
    parser.add_argument('--index_input_dir', type=str, required=True,
                        help='GraphRAG索引的input目录路径')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='检查前N个样本的文档覆盖情况（None表示检查全部）')

    args = parser.parse_args()

    print("="*80)
    print("GraphRAG索引覆盖度检查")
    print("="*80)
    print(f"测试文件: {args.test_file}")
    print(f"索引input目录: {args.index_input_dir}")
    print(f"检查样本数: {args.num_samples if args.num_samples else '全部'}")
    print("="*80)

    # 1. 提取测试集所需的文档标题
    print("\n[1/3] 提取测试集文档标题...")
    test_titles = extract_test_titles(args.test_file, args.num_samples)
    print(f"  测试集共需 {len(test_titles)} 个文档")

    # 2. 获取索引中的文档标题
    print("\n[2/3] 读取索引文档标题...")
    index_docs = get_index_docs(args.index_input_dir)
    print(f"  索引包含 {len(index_docs)} 个文档")

    # 3. 检查覆盖情况
    print("\n[3/3] 分析覆盖情况...")
    coverage = check_coverage(test_titles, index_docs)

    # 输出统计结果
    print("\n" + "="*80)
    print("覆盖度统计结果")
    print("="*80)
    print(f"测试集所需文档总数:  {coverage['test_total']}")
    print(f"索引包含文档总数:    {coverage['index_total']}")
    print(f"匹配的文档数:        {coverage['matched']} ({coverage['coverage_rate']*100:.2f}%)")
    print(f"缺失的文档数:        {coverage['missing']}")
    print(f"多余的文档数:        {coverage['extra']}")

    # 显示缺失的文档（最多显示20个）
    if coverage['missing'] > 0:
        print(f"\n缺失的文档（显示前20个，共{coverage['missing']}个）:")
        for title in coverage['missing_titles'][:20]:
            print(f"  - {title}")
        if coverage['missing'] > 20:
            print(f"  ... 还有 {coverage['missing'] - 20} 个")

    # 显示匹配的文档（最多显示20个）
    if coverage['matched'] > 0:
        print(f"\n匹配的文档（显示前20个，共{coverage['matched']}个）:")
        for title in coverage['matched_titles'][:20]:
            print(f"  [OK] {title}")
        if coverage['matched'] > 20:
            print(f"  ... 还有 {coverage['matched'] - 20} 个")

    # 结论
    print("\n" + "="*80)
    if coverage['coverage_rate'] >= 0.8:
        print("[OK] 索引覆盖度高 (>=80%)，测试集文档基本齐全")
    elif coverage['coverage_rate'] >= 0.5:
        print("[WARN] 索引覆盖率中等 (50%-80%)，部分文档缺失")
    else:
        print("[ERROR] 索引覆盖率低 (<50%)，可能严重影响性能")
    print("="*80)

if __name__ == '__main__':
    main()
