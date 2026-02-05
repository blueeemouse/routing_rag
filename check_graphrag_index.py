"""检查GraphRAG索引的质量"""
import json
import os
from pathlib import Path
import glob

# 切换到脚本所在目录
script_dir = Path(__file__).parent
os.chdir(script_dir)

def analyze_index(index_path, name):
    """分析索引的质量指标"""
    print(f"\n{'='*60}")
    print(f"{name} 索引分析")
    print(f"{'='*60}")

    # 检查索引文件
    if not os.path.exists(index_path):
        print(f"索引目录不存在: {index_path}")
        return

    # 统计文件数量和大小
    parquet_files = glob.glob(os.path.join(index_path, "*.parquet"))
    lance_files = glob.glob(os.path.join(index_path, "*.lance"))
    all_files = glob.glob(os.path.join(index_path, "*.*"))

    print(f"\n索引文件统计:")
    print(f"  总文件数: {len(all_files)}")
    print(f"  Parquet文件数: {len(parquet_files)}")
    print(f"  Lance文件数: {len(lance_files)}")

    if parquet_files:
        total_size = sum(os.path.getsize(f) for f in parquet_files) / (1024*1024)
        print(f"  Parquet总大小: {total_size:.2f} MB")

    # 检查关键文件
    print(f"\n关键文件:")
    key_files = ['documents.parquet', 'entities.parquet', 'relationships.parquet']
    for kf in key_files:
        path = os.path.join(index_path, kf)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"  {kf}: {size:.2f} MB")

    # 尝试读取统计信息
    for parquet_file in ['documents.parquet', 'entities.parquet']:
        path = os.path.join(index_path, parquet_file)
        if os.path.exists(path):
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                print(f"\n{parquet_file}:")
                print(f"  行数: {len(df)}")
                print(f"  列数: {len(df.columns)}")
                print(f"  列名: {list(df.columns)[:5]}...")
            except Exception as e:
                print(f"\n无法读取 {parquet_file}: {e}")

# 检查两个索引
index_1000 = "naive_rag_index_hotpotqa_1000_samples"
index_dev = "naive_rag_index_hotpotqa_dev_1000_samples"

# 尝试找到GraphRAG索引
possible_graphrag_indices = [
    "graphrag_index_hotpotqa_train_5000_samples",
    "graphrag_index_hotpotqa_1000_samples",
    "graphrag_ollama_hotpotqa_test_data",
    "graphrag_ollama_hotpotqa_1000_test_data",
]

for idx_path in possible_graphrag_indices:
    if os.path.exists(idx_path):
        analyze_index(idx_path, f"GraphRAG索引: {idx_path}")

# 用户需要指定实际使用的GraphRAG索引路径
print(f"\n{'='*60}")
print(f"请确认实际使用的GraphRAG索引路径:")
print(f"{'='*60}")
print("根据你的实验，应该有一个基于 dev_distractor_1000 的索引")
print("和一个基于 hotpot_1000 的索引")
print("\n请在代码中指定正确的索引路径")
