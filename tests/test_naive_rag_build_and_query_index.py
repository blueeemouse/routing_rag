"""
从 HopRAG 提供的 1000 样本 hotpot.jsonl 文件中提取前 N 个样本的文档，
用于构建 Naive RAG 索引的初步测试。
"""

import json
import sys
import os

# Add parent directory to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from config.config import settings
# Import LlamaIndex components that support saving/loading
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
import os

def extract_sample_docs_from_jsonl(jsonl_file_path, num_samples=50):
    """
    从 JSONL 文件中提取前 N 个样本的文档文本块。

    Args:
        jsonl_file_path (str): JSONL 文件路径。
        num_samples (int): 要提取的样本数量。

    Returns:
        list: 包含文档文本的列表。
    """
    documents = []
    print(f"正在从 {jsonl_file_path} 读取前 {num_samples} 个样本...")
    count = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= num_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                # 提取 context
                context = data.get('context', [])
                for title, sentence_list in context:
                    # 按照 HopRAG 的方式连接句子
                    doc_text = "\n\n".join(sentence_list)
                    documents.append(doc_text)
                count += 1
                if count % 10 == 0: # 每处理10个样本打印一次进度
                    print(f"  已处理 {count}/{num_samples} 个样本...")

            except json.JSONDecodeError as e:
                print(f"警告: 解析第 {count+1} 行 JSON 时出错: {e}")
                continue
            except Exception as e:
                print(f"警告: 处理第 {count+1} 行时出错: {e}")
                continue

    print(f"文档提取完成，共处理了 {count} 个样本，获得 {len(documents)} 个文档块。")
    return documents

def save_index_to_disk(index, storage_dir="./naive_rag_index_storage"):
    """将 LlamaIndex 索引保存到磁盘。"""
    print(f"正在将索引保存到 {storage_dir} ...")
    os.makedirs(storage_dir, exist_ok=True)
    # storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index.storage_context.persist(persist_dir=storage_dir)
    print("索引保存完成。")

def load_index_from_disk(storage_dir="./naive_rag_index_storage"):
    """从磁盘加载 LlamaIndex 索引。"""
    print(f"正在从 {storage_dir} 加载索引 ...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print("索引加载完成。")
        return index
    except Exception as e:
        print(f"加载索引失败: {e}")
        return None

if __name__ == "__main__":
    # 1. 从 HopRAG 的 1000 样本 JSONL 中提取文档
    hotpot_jsonl_path = os.path.join(PARENT_DIR, "HotpotQA", "hotpot_1000_samples.jsonl")
    num_samples_to_use = 60 # 先测试 50 条
    # num_samples_to_use = 1000
    sample_docs_data_list = extract_sample_docs_from_jsonl(hotpot_jsonl_path, num_samples=num_samples_to_use)

    if not sample_docs_data_list:
        print("未能提取到文档数据，退出。")
        exit()

    print(f"准备用前 {num_samples_to_use} 个样本的 {len(sample_docs_data_list)} 个文档块构建 Naive RAG 索引。")

    # 2. 初始化 NaiveRAG 实例
    print("正在初始化 NaiveRAG 实例...")
    naive_rag_instance = NaiveRAG()

    # 3. 构建索引
    print("正在构建 LlamaIndex Naive RAG 索引...")
    success = naive_rag_instance.build_index_from_data(sample_docs_data_list)

    if success:
    # if True:
        print("LlamaIndex Naive RAG 索引构建成功！")
        print(f"索引中包含 {len(naive_rag_instance.documents)} 个文档。")

        # 4. 尝试保存索引
        STORAGE_DIR = os.path.join(PARENT_DIR, f"naive_rag_index_storage_{num_samples_to_use}_samples") # 为不同采样数创建不同目录
        save_index_to_disk(naive_rag_instance.index, storage_dir=STORAGE_DIR)

        # 5. 尝试加载索引 (验证保存/加载功能)
        print("\n--- 测试加载已保存的索引 ---")
        loaded_index = load_index_from_disk(storage_dir=STORAGE_DIR)
        if loaded_index is not None:
            print("索引加载成功，可以用于后续查询。")
            # 可以在此处实例化一个新的 NaiveRAG 对象，并手动设置其 index 属性为 loaded_index
            # 或者修改 NaiveRAG 类，添加 load_index_from_storage 方法
            # 关键步骤：将加载的索引赋值给 NaiveRAG 实例的 index 属性
            naive_rag_instance.index = loaded_index
            naive_rag_instance.is_index_initialized = True
            
            # test_query = "What is Major League Soccer?"
            test_query = "Donnie Smith who plays as a left back for New England Revolution belongs to what league featuring 22 teams?"
            print(f"执行测试查询: '{test_query}'")
            try:
                response = naive_rag_instance.execute(test_query) # 使用原实例 (它已构建索引)
                print(f"响应: {response[:100]}...") # 打印前100个字符
            except Exception as e:
                print(f"执行查询时出错: {e}")
        else:
            print("索引加载失败。")

    else:
        print("LlamaIndex Naive RAG 索引构建失败！")
        print("请检查您的 API 配置是否正确。")