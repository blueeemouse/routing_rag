"""
NaiveRAG 索引构建脚本
从 HotpotQA 数据构建 NaiveRAG 索引

使用示例:
    # 基本用法
    python build_naive_rag_index.py --hotpotqa_file "data.jsonl" --index_path "D:\\index_storage"

    # 指定样本数量
    python build_naive_rag_index.py --hotpotqa_file "data.jsonl" --index_path "D:\\index_storage" --num_samples 5000
"""

import argparse
import sys
import os
import json

# Add routing_rag path
ROUTING_RAG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROUTING_RAG_ROOT)

# 加载环境变量
ENV_FILE = os.path.join(ROUTING_RAG_ROOT, '.env')
if os.path.exists(ENV_FILE):
    print(f"正在加载环境变量: {ENV_FILE}")
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print("环境变量加载完成")
else:
    print(f"警告: 未找到.env文件: {ENV_FILE}")

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG


def load_hotpotqa_documents(hotpotqa_file: str, num_samples: int = None) -> list:
    """
    从 HotpotQA 数据文件加载文档

    Args:
        hotpotqa_file: HotpotQA 数据文件路径
        num_samples: 处理的样本数量（None 表示处理全部）

    Returns:
        documents: 文档文本列表
    """
    documents = []
    count = 0

    print(f"\n正在从 HotpotQA 数据加载文档...")
    print(f"输入文件: {hotpotqa_file}")
    if num_samples:
        print(f"样本数量: {num_samples}")

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

                # 提取每个文档
                for title, sentence_list in context:
                    doc_text = "\n\n".join(sentence_list)
                    documents.append(doc_text)

                count += 1
                if count % 500 == 0:
                    print(f"  已处理 {count}/{num_samples if num_samples else 'all'} 个样本，提取 {len(documents)} 个文档...")

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue

    print(f"文档加载完成！共处理 {count} 个样本，提取 {len(documents)} 个文档。")
    return documents


def main():
    parser = argparse.ArgumentParser(
        description='NaiveRAG 索引构建脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基本用法
    python build_naive_rag_index.py --hotpotqa_file "data.jsonl" --index_path "D:\\index_storage"

    # 指定样本数量
    python build_naive_rag_index.py --hotpotqa_file "data.jsonl" --index_path "D:\\index_storage" --num_samples 5000

    # 加载已有索引（如果存在）
    python build_naive_rag_index.py --hotpotqa_file "data.jsonl" --index_path "D:\\index_storage" --load_existing
        """
    )

    parser.add_argument(
        '--hotpotqa_file',
        type=str,
        required=True,
        help='HotpotQA 数据文件路径'
    )

    parser.add_argument(
        '--index_path',
        type=str,
        required=True,
        help='NaiveRAG 索引存储路径'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='处理的样本数量（可选，默认处理全部）'
    )

    parser.add_argument(
        '--load_existing',
        action='store_true',
        help='如果索引已存在，尝试加载而不是重新构建'
    )

    args = parser.parse_args()

    # 验证输入文件
    if not os.path.exists(args.hotpotqa_file):
        print(f"错误: HotpotQA 数据文件不存在: {args.hotpotqa_file}")
        return 1

    # 构建完整路径
    index_path = os.path.abspath(args.index_path)

    # 打印配置信息
    print("\n" + "=" * 60)
    print("NaiveRAG 索引构建")
    print("=" * 60)
    print(f"数据文件: {args.hotpotqa_file}")
    print(f"索引路径: {index_path}")
    if args.num_samples:
        print(f"样本数量: {args.num_samples}")
    print(f"加载已有索引: {'是' if args.load_existing else '否'}")
    print("=" * 60 + "\n")

    # 创建 NaiveRAG 实例
    naive_rag = NaiveRAG()

    # 尝试加载已有索引
    if args.load_existing and os.path.exists(index_path) and os.listdir(index_path):
        print(f"尝试加载已有索引: {index_path}")
        if naive_rag.load_index(index_path):
            print("索引加载成功！")
            print(f"索引包含 {len(naive_rag.documents)} 个文档")
            return 0
        else:
            print("索引加载失败，将重新构建...")

    # 加载文档
    documents = load_hotpotqa_documents(args.hotpotqa_file, args.num_samples)

    if not documents:
        print("错误: 未能加载任何文档")
        return 1

    # 构建索引
    print("\n开始构建 NaiveRAG 索引...")
    success = naive_rag.build_index_from_data(documents)

    if not success:
        print("索引构建失败！")
        return 1

    print(f"索引构建成功，包含 {len(naive_rag.documents)} 个文档")

    # 保存索引
    os.makedirs(index_path, exist_ok=True)
    print(f"\n保存索引到: {index_path}")
    save_success = naive_rag.save_index(index_path)

    if save_success:
        print("\n" + "=" * 60)
        print("NaiveRAG 索引构建并保存成功！")
        print("=" * 60)
        return 0
    else:
        print("索引保存失败！")
        return 1


if __name__ == '__main__':
    sys.exit(main())
