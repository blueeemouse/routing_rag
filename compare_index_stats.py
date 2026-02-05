"""对比两个GraphRAG索引的关键统计信息"""
import pandas as pd
import json
import os
from pathlib import Path

# 切换到脚本所在目录
script_dir = Path(__file__).parent
os.chdir(script_dir)

print("="*80)
print("GraphRAG 索引统计对比")
print("="*80)

# 15条索引
index_15 = "graphrag_ollama_hotpotqa_test_data/output"
index_1000 = "graphrag_ollama_hotpotqa_1000_test_data/output"

# 1. 统计实体
print("\n1. 实体统计")
print("-"*80)
entities_15 = pd.read_parquet(f"{index_15}/entities.parquet")
entities_1000 = pd.read_parquet(f"{index_1000}/entities.parquet")

print(f"{'指标':<30} {'15条索引':<15} {'1000条索引':<15}")
print(f"{'实体数量':<30} {len(entities_15):>10} {len(entities_1000):>10}")

# 检查 description 列
desc_empty_15 = entities_15['description'].isna().sum()
desc_empty_1000 = entities_1000['description'].isna().sum()
desc_valid_15 = len(entities_15) - desc_empty_15
desc_valid_1000 = len(entities_1000) - desc_empty_1000

print(f"有description的实体: {desc_valid_15:>10} {desc_valid_1000:>10}")
print(f"无description的实体: {desc_empty_15:>10} {desc_empty_1000:>10}")

if desc_valid_15 > 0:
    avg_desc_len_15 = entities_15[entities_15['description'].notna()]['description'].str.len().mean()
else:
    avg_desc_len_15 = 0

if desc_valid_1000 > 0:
    avg_desc_len_1000 = entities_1000[entities_1000['description'].notna()]['description'].str.len().mean()
else:
    avg_desc_len_1000 = 0

print(f"平均description长度: {avg_desc_len_15:>10.1f} {avg_desc_len_1000:>10.1f}")

# 2. 统计关系
print("\n2. 关系统计")
print("-"*80)
rel_15 = pd.read_parquet(f"{index_15}/relationships.parquet")
rel_1000 = pd.read_parquet(f"{index_1000}/relationships.parquet")

print(f"{'指标':<30} {'15条索引':<15} {'1000条索引':<15}")
print(f"{'关系数量':<30} {len(rel_15):>10} {len(rel_1000):>10}")

# 3. 统计社区报告
print("\n3. 社区报告统计")
print("-"*80)
comm_15 = pd.read_parquet(f"{index_15}/community_reports.parquet")
comm_1000 = pd.read_parquet(f"{index_1000}/community_reports.parquet")

print(f"{'指标':<30} {'15条索引':<15} {'1000条索引':<15}")
print(f"{'社区数量':<30} {len(comm_15):>10} {len(comm_1000):>10}")

# 检查 community 的 full_content
if 'full_content' in comm_15.columns:
    comm_content_15 = comm_15['full_content'].str.len().mean()
else:
    comm_content_15 = 0

if 'full_content' in comm_1000.columns:
    comm_content_1000 = comm_1000['full_content'].str.len().mean()
else:
    comm_content_1000 = 0

print(f"平均社区报告长度: {comm_content_15:>10.1f} {comm_content_1000:>10.1f}")

# 4. 统计文本单元
print("\n4. 文本单元统计")
print("-"*80)
tu_15 = pd.read_parquet(f"{index_15}/text_units.parquet")
tu_1000 = pd.read_parquet(f"{index_1000}/text_units.parquet")

print(f"{'指标':<30} {'15条索引':<15} {'1000条索引':<15}")
print(f"{'文本单元数量':<30} {len(tu_15):>10} {len(tu_1000):>10}")

tu_len_15 = tu_15['text'].str.len().mean()
tu_len_1000 = tu_1000['text'].str.len().mean()
print(f"平均文本单元长度: {tu_len_15:>10.1f} {tu_len_1000:>10.1f}")

# 5. 查看 stats.json
print("\n5. 整体统计")
print("-"*80)

with open(f"{index_15}/stats.json", 'r', encoding='utf-8') as f:
    stats_15 = json.load(f)

with open(f"{index_1000}/stats.json", 'r', encoding='utf-8') as f:
    stats_1000 = json.load(f)

print(f"{'指标':<30} {'15条索引':<15} {'1000条索引':<15}")

if 'summarized_communities_count' in stats_15:
    print(f"摘要的社区数量: {stats_15.get('summarized_communities_count', 'N/A'):>10} {stats_1000.get('summarized_communities_count', 'N/A'):>10}")

# 6. 实体类型分布
print("\n6. 实体类型分布 (Top 10)")
print("-"*80)

if 'type' in entities_15.columns:
    type_counts_15 = entities_15['type'].value_counts().head(10)
    print("15条索引:")
    for entity_type, count in type_counts_15.items():
        print(f"  {entity_type}: {count}")

if 'type' in entities_1000.columns:
    type_counts_1000 = entities_1000['type'].value_counts().head(10)
    print("\n1000条索引:")
    for entity_type, count in type_counts_1000.items():
        print(f"  {entity_type}: {count}")

print("\n" + "="*80)
print("关键观察点:")
print("="*80)
print("""
1. 1000条索引的description是否大量为空？
   → 如果是，说明实体提取有问题

2. 1000条索引的社区数量是否异常？
   → 15条数据可能有50个社区，1000条可能有500+个社区
   → 社区太多可能导致检索不精确

3. 实体类型分布是否正常？
   → 应该有 person, organization, location 等

4. 文本单元长度是否合理？
   → 应该在合理范围内（几百字符）

如果发现异常：
→ 需要检查索引构建时的日志（如果有）
→ 或者查看GraphRAG的社区检测参数
""")
