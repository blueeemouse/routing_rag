# GraphRAG性能异常分析报告

## 问题现象

| 模型 | Exact Match (EM) | F1 Score | Precision | Recall | 说明 |
|------|-----------------|----------|-----------|---------|------|
| NoRAG | 12.5% | 17.9% | 19.5% | 17.3% | 直接使用LLM先验知识 |
| NaiveRAG | 35.8% | 46.7% | 49.9% | 46.1% | 向量检索+LLM |
| **GraphRAG** | **9.6%** | **13.8%** | **15.0%** | **13.6%** | 图检索+LLM |

**关键问题**：GraphRAG性能甚至低于NoRAG（没有检索），这不合理！

---

## 已添加的调试日志

已在 `graph_rag_impl.py` 的 `_local_search` 方法中添加详细日志输出（第625-680行）：

1. 查询信息（query文本）
2. 图数据统计（实体、关系、社区报告、文本单元数量）
3. 检索到的context信息：
   - context_data数量
   - 前3条context记录的详细信息
   - 拼接的context文本（前500字符）
4. LLM调用统计
5. 最终答案

---

## 初步发现

### 1. 答案模式分析

从评测结果中观察到：

**GraphRAG的答案类型**：
- 具体答案：Governor, New York City, SM Entertainment, Tony Hsieh, Rock Chalk Jayhawk等
- "I don't know"：部分问题返回此答案

**NoRAG的答案**：
- 具体答案：no, Governor, Eidolon, New York City, SM Entertainment, Jack Stack等
- 答案与GraphRAG**高度相似**，很多完全一致！

**NaiveRAG的答案**：
- 大量返回"I don't know"（检索不到相关文档）
- 检索到时给出正确答案（如YG Entertainment）

**关键发现**：
- GraphRAG和NoRAG的答案高度重合，说明GraphRAG可能**没有检索到有效context**
- LLM在没有context时，会像NoRAG一样使用先验知识

### 2. 可能的原因

#### 原因1：向量存储连接问题 ⭐⭐⭐（最可能）

**问题**：
```python
# graph_rag_impl.py:557-564
schema_config = VectorStoreSchemaConfig(
    vector_size=1536,  # ← 硬编码，可能与实际维度不匹配
)
```

虽然 `similarity_search_by_vector` 不依赖这个参数，但如果构建索引时的维度和查询时不一致：
- 可能在向量检索时返回空结果
- 或者返回错误的相关实体

**验证方法**：
运行带有调试日志的查询，查看：
- 是否检索到实体（context_data数量）
- 检索到的实体是否相关

#### 原因2：索引构建时使用的embedding模型与查询时不一致 ⭐⭐

**配置文件**（graphrag_hotpotqa_config.yml）：
```yaml
models:
  default_embedding_model:
    model: nomic-embed-text  # ← 768维
```

**查询时**：
- 使用相同的配置，应该是相同的模型
- 但需要确认索引确实是用 `nomic-embed-text` 构建的

#### 原因3：HotpotQA数据不适合GraphRAG ⭐

**特点**：
- HotpotQA是多跳推理任务
- 需要跨多个文档推理
- GraphRAG的社区报告可能无法捕捉细粒度的关系

**但这不能解释为什么比NoRAG还差！**

#### 原因4：Context构建参数配置不当 ⭐⭐

**需要检查的配置**：
- `top_k_entities`: 检索的实体数量
- `max_context_tokens`: 最大context token数
- `community_prop`: 社区报告权重

如果这些参数过小：
- 可能检索不到足够的实体
- 导致context信息不足

---

## 建议的调试步骤

### 步骤1：运行单条查询测试（带详细日志）

```bash
cd D:\Develop\all_RAG\routing_rag
python debug_graphrag.py
```

**预期输出**：
```
================================================================================
GraphRAG查询调试信息
================================================================================
查询: Were Scott Derrickson and Ed Wood of the same nationality?
--------------------------------------------------------------------------------
实体数量: XXXX
关系数量: XXXX
社区报告数量: XXXX
文本单元数量: XXXX

--- 检索到的context信息 ---
context_data数量: 0
⚠️  没有检索到任何context_data！

--- 拼接的context文本 (前500字符) ---
[空或无关内容]

--- 最终答案 (完整) ---
No
================================================================================
```

**关键指标**：
- 如果 `context_data数量 = 0`，说明向量检索完全失败！
- 如果 `context_data数量 > 0` 但内容不相关，说明向量检索到错误内容

### 步骤2：检查向量存储的实际情况

**方法A：使用Python脚本检查**

```python
import pandas as pd
from pathlib import Path

lancedb_path = Path("D:/Develop/all_RAG/routing_rag/graphrag_ollama_hotpotqa_1000_test_data/output/lancedb")

# 查看lancedb目录下的文件
print("LanceDB文件：")
for f in lancedb_path.glob("*"):
    print(f"  {f.name}")
```

**方法B：查看Parquet文件**

```python
import pandas as pd

entities = pd.read_parquet("D:/Develop/all_RAG/routing_rag/graphrag_ollama_hotpotqa_1000_test_data/output/entities.parquet")
print(f"实体数量: {len(entities)}")
print(f"实体列: {entities.columns.tolist()}")
if 'description_embeddings' in entities.columns:
    embeddings = entities['description_embeddings'].dropna()
    print(f"有嵌入的实体数量: {len(embeddings)}")
    if len(embeddings) > 0:
        print(f"嵌入向量维度: {len(embeddings.iloc[0]['description_embeddings'])}")
```

### 步骤3：重新构建索引（如果必要）

**如果索引有问题，重新构建**：

```bash
python .\tests\unified_evaluate_hotpotqa.py `
    --models graph_rag `
    --graphrag_work_dir "D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data" `
    --num_samples 100 `
    --delay 2.5
    # 注意：不带 --skip_graphrag_index 参数
```

**这将重新构建索引，确保一致性**

---

## 下一步行动

1. ✅ 已添加调试日志到 `graph_rag_impl.py`
2. ⏳ 需要运行测试查看实际检索的context
3. ⏳ 根据日志输出确定具体问题
4. ⏳ 修复根本原因
5. ⏳ 重新评测验证性能提升

---

## 待确认的问题

1. 向量维度配置是否正确（768 vs 1536）？
2. 向量存储是否正确加载？
3. 查询时使用的embedding模型是否与构建索引时一致？
4. LocalSearch的参数配置是否合理？

请运行 `debug_graphrag.py` 并分享输出结果，我们可以进一步诊断问题！
