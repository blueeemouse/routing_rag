"""
GraphRAG 索引构建脚本
支持 Standard 和 Fast 两种索引构建模式

使用示例:
    # Standard 模式（默认）
    python build_graphrag_index.py --work_dir "D:\path\to\work_dir" --config_file "config.yml"

    # Fast 模式
    python build_graphrag_index.py --work_dir "D:\path\to\work_dir" --config_file "config.yml" --method fast

    # 从 HotpotQA 数据准备输入并构建索引
    python build_graphrag_index.py --work_dir "D:\path\to\work_dir" --config_file "config.yml" --method fast --hotpotqa_file "data.jsonl" --num_samples 5000
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

from rag_implementations.graph_rag.graph_rag_impl import GraphRAG


def prepare_hotpotqa_data_for_graphrag(hotpotqa_file: str, output_dir: str, num_samples: int = None):
    """
    准备HotpotQA数据用于GraphRAG索引构建
    将HotpotQA的文档提取为文本文件

    Args:
        hotpotqa_file: HotpotQA数据文件路径
        output_dir: 输出目录
        num_samples: 处理的样本数量（None表示处理全部）
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n正在从HotpotQA数据中提取文档...")
    print(f"输入文件: {hotpotqa_file}")
    print(f"输出目录: {output_dir}")
    if num_samples:
        print(f"样本数量: {num_samples}")

    count = 0
    doc_count = 0

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
                for title, sentences in context:
                    doc_text = "\n\n".join(sentences)

                    # 保存为文本文件，使用安全的文件名
                    safe_title = title.replace('/', '_').replace('\\', '_').replace(':', '_')
                    for char in '<>:"\\|?*':
                        safe_title = safe_title.replace(char, '_')
                    doc_file = os.path.join(output_dir, f"{safe_title}.txt")

                    with open(doc_file, 'w', encoding='utf-8') as f:
                        f.write(f"{title}\n\n{doc_text}")

                    doc_count += 1

                count += 1
                if count % 100 == 0:
                    print(f"  已处理 {count}/{num_samples if num_samples else 'all'} 个样本，提取 {doc_count} 个文档...")

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue

    print(f"文档提取完成！共处理 {count} 个样本，提取 {doc_count} 个文档。")
    return doc_count


def main():
    parser = argparse.ArgumentParser(
        description='GraphRAG 索引构建脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 使用现有配置文件构建索引
    python build_graphrag_index.py --work_dir "D:\\work_dir" --config_file "config.yml"

    # Fast 模式
    python build_graphrag_index.py --work_dir "D:\\work_dir" --config_file "config.yml" --method fast

    # 从 HotpotQA 数据准备输入并构建索引
    python build_graphrag_index.py --work_dir "D:\\work_dir" --config_file "config.yml" --method fast --hotpotqa_file "data.jsonl" --num_samples 5000

索引方法说明:
    - standard: 使用 LLM 进行完整的图构建，图质量高但速度慢
    - fast: 使用 NLP + LLM 混合模式，速度快但图精度略低
        """
    )

    parser.add_argument(
        '--work_dir',
        type=str,
        required=True,
        help='GraphRAG 工作目录'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='GraphRAG 配置文件名（相对于 work_dir）'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='standard',
        choices=['standard', 'fast'],
        help='索引构建方法: standard (默认) 或 fast'
    )

    parser.add_argument(
        '--hotpotqa_file',
        type=str,
        default=None,
        help='HotpotQA 数据文件路径（可选，用于准备输入数据）'
    )

    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='处理的样本数量（可选，配合 --hotpotqa_file 使用）'
    )

    args = parser.parse_args()

    # 构建完整路径
    work_dir = os.path.abspath(args.work_dir)
    config_path = os.path.join(work_dir, args.config_file)

    # 验证工作目录
    if not os.path.exists(work_dir):
        print(f"错误: 工作目录不存在: {work_dir}")
        return 1

    # 验证配置文件（必须存在）
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return 1
    else:
        print(f"配置文件已存在: {config_path}")

    # 准备输入数据
    if args.hotpotqa_file:
        input_dir = os.path.join(work_dir, 'input')
        prepare_hotpotqa_data_for_graphrag(args.hotpotqa_file, input_dir, args.num_samples)
    else:
        input_dir = os.path.join(work_dir, 'input')
        if not os.path.exists(input_dir) or not os.listdir(input_dir):
            print(f"警告: 输入目录为空或不存在: {input_dir}")
            print("请使用 --hotpotqa_file 参数准备输入数据，或手动准备 input 目录")

    # 打印配置信息
    print("\n" + "=" * 60)
    print("GraphRAG 索引构建")
    print("=" * 60)
    print(f"工作目录: {work_dir}")
    print(f"配置文件: {config_path}")
    print(f"构建方法: {args.method.upper()}")
    print("=" * 60 + "\n")

    # 创建 GraphRAG 实例并构建索引
    graphrag = GraphRAG()

    success = graphrag.build_index_from_path(
        root_dir=work_dir,
        config_filepath=config_path,
        method=args.method
    )

    if success:
        print("\n" + "=" * 60)
        print("GraphRAG 索引构建成功！")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("GraphRAG 索引构建失败！")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
