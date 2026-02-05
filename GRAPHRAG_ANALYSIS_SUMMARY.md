# GraphRAG 性能问题 - 源码分析总结

## 关键发现

### 1. `embed_graph.enabled` 的实际作用

**位置**: `graphrag/index/operations/embed_graph/embed_graph.py`

```python
def embed_graph(
    graph: nx.Graph,
    config: EmbedGraphConfig,
) -> NodeEmbeddings:
    """
    Embed a graph into a vector space using node2vec.
    """
    if config.use_lcc:
        graph = stable_largest_connected_component(graph)

    # create graph embedding using node2vec
    embeddings = embed_node2vec(
        graph=graph,
        dimensions=config.dimensions,
        num_walks=config.num_walks,
        walk_length=config.walk_length,
        window_size=config.window_size,
        iterations=config.iterations,
        random_seed=config.random_seed,
    )

    pairs = zip(embeddings.nodes, embeddings.embeddings.tolist(), strict=True)
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    return dict(sorted_pairs)
```

**作用**: 使用 **node2vec** 对图谱进行向量化嵌入

### 2. LocalSearch 实际使用的 embeddings

**位置**: `graphrag/query/context_builder/entity_extraction.py`

```python
def map_query_to_entities(
    query: str,
    text_embedding_vectorstore: BaseVectorStore,  # ← 使用的是 text embedding
    text_embedder: EmbeddingModel,
    ...
) -> list[Entity]:
    """Extract entities that match a given query using semantic similarity
    of text embeddings of query and entity descriptions."""

    search_results = text_embedding_vectorstore.similarity_search_by_text(
        text=query,
        text_embedder=lambda t: text_embedder.embed(t),  # ← 用text embedder
        k=k * oversample_scaler,
    )
```

**关键**: LocalSearch 使用的是 **entity 的 description embedding**，这是从 **text embedding** 生成的，不是从 **graph embedding** 生成的！

### 3. Entity Description Embedding 的生成

**位置**: `graphrag/index/workflows/generate_text_embeddings.py`

```python
entity_description_embedding: {
    "data": entities.loc[:, ["id", "title", "description"]].assign(
        title_description=lambda df: df["title"] + ":" + df["description"]
    )
    if entities is not None
    else None,
    "embed_column": "title_description",
},
```

**流程**:
1. 提取实体（title + description）
2. 用 text embedding model (nomic-embed-text) 对 `title_description` 进行嵌入
3. 存储到 vector store 供检索使用

### 4. 配置关系

```
embed_text (Text Embedding)
  ├─ model: nomic-embed-text
  ├─ names: [entity_description_embedding, community_full_content_embedding, ...]
  └─ 用于检索 (LocalSearch)

embed_graph (Graph Embedding)
  ├─ enabled: true/false
  ├─ 使用 node2vec
  └─ 用于布局 和全局搜索
```

## 结论

### `embed_graph.enabled` 可能不是主要问题

**原因**:
- LocalSearch 使用的是 **entity description embedding**（来自 text embedding）
- 这个 embedding 的生成与 `embed_graph.enabled` 无关
- 只要在 `embed_text.names` 中包含 `entity_description_embedding`，就会生成

### 真正需要检查的配置

#### 1. `embed_text.names` 是否配置正确？

两个配置文件都没有显式设置 `embed_text.names`，应该使用默认值：

```python
default_embeddings: list[str] = [
    entity_description_embedding,  # ← 默认启用
    community_full_content_embedding,
]
```

#### 2. 其他关键配置差异

| 配置项 | 15条索引 | 1000条索引 | 影响 |
|---------|-----------|-------------|------|
| `embed_graph.enabled` | true | false | 布局、全局搜索 |
| `concurrent_requests` | 25 | 16 | 并发性能 |
| `request_timeout` | 180.0 | 1200.0 | 超时时间 |

#### 3. 检索参数

需要检查 `local_search` 中的参数：
- `top_k`: 检索的实体数量
- `max_context_tokens`: 最大上下文长度
- `num_tokens`: 文本单元大小

这些参数是否在两个配置中一致？

## 建议的下一步

### 第1步：检查检索参数

在两个配置文件中查找 `local_search` 部分，对比：
- top_k
- max_context_tokens
- 使用的是哪个 embedding_model_id

### 第2步：查看实际检索日志

运行 GraphRAG 并记录：
- 每个问题检索到的实体数量
- 检索到的实体名称
- 最终上下文长度

### 第3步：手动对比3个重叠问题

```bash
python compare_retrieval.py
```

选择3个重叠问题，在两种索引下测试，对比：
1. 检索到的实体是否相关
2. 是否包含答案线索
3. 上下文质量

### 第4步：检查社区检测

两个索引的社区检测结果可能不同：
- 15条：50个文档 → 小社区结构
- 1000条：9766个文档 → 大社区结构

这可能导致：
- 大索引时社区太大，社区报告不够精确
- 或者社区检测算法在大规模数据上有bug

## 可能的根本原因（按概率）

1. **检索参数不一致 (40%)** - top_k、max_tokens 等参数不同
2. **社区检测问题 (30%)** - 大规模数据时社区检测效果差
3. **实体描述质量 (20%)** - 大规模索引时实体提取/描述质量差
4. **其他未知问题 (10%)**

## 快速验证方法

### 方法1：检查日志文件

```bash
# 查看索引构建日志
grep -i "error" graphrag_ollama_hotpotqa_1000_test_data/logs/*.log
grep -i "warning" graphrag_ollama_hotpotqa_1000_test_data/logs/*.log
```

### 方法2：对比两个索引的实体数量

使用 `check_graphrag_index.py`（已创建）

### 方法3：查看 parquet 文件

```python
import pandas as pd

entities_15 = pd.read_parquet('graphrag_ollama_hotpotqa_test_data/output/entities.parquet')
entities_1000 = pd.read_parquet('graphrag_ollama_hotpotqa_1000_test_data/output/entities.parquet')

print(f"15条索引实体数: {len(entities_15)}")
print(f"1000条索引实体数: {len(entities_1000)}")

# 检查 description 列
print(f"\n15条索引 description 为空的实体: {entities_15['description'].isna().sum()}")
print(f"1000条索引 description 为空的实体: {entities_1000['description'].isna().sum()}")
```
