# 图RAG实现 (Graph RAG Implementation)

此模块实现图RAG策略，基于微软GraphRAG框架。

## 功能
- 基于知识图谱的RAG实现
- 图结构检索和生成流程
- 提供统一的RAG接口
- 支持通过API配置LLM和嵌入模型
- 需要完整的图数据结构（实体、关系、社区等）才能正常工作

## 配置系统

### 配置文件说明

完整的配置参数结构请参考：`config/settings_template.yaml`

该模板文件展示了所有可配置参数及其层级关系，特别是 `embedding_dim` 和 `embedding_model` 的同级关系。

### 配置优先级

GraphRAG 实现支持灵活的配置管理，配置参数优先级如下：

**初始化配置 > 全局配置（settings.yaml）**

```python
# 方式1：使用全局配置
graph_rag = GraphRAG()

# 方式2：使用自定义配置（覆盖全局配置）
custom_config = {
    'api_url': 'http://localhost:11434/v1',
    'api_key': 'ollama',
    'model': 'llama3.2',
    'embedding_model': 'nomic-embed-text',
    'embedding_dim': 768,
    'top_k': 10
}
graph_rag = GraphRAG(config=custom_config)
```

### 向量维度配置

向量维度确定采用严格匹配策略，优先级如下：

1. **显式配置维度**：如果配置中指定了 `embedding_dim`，直接使用该值
2. **模型名推断**：如果没有显式配置，从 `embedding_model` 推断维度
3. **严格报错**：如果既无显式配置，模型又不在已知列表中，则报错并退出

#### 已知的 Embedding 模型维度映射

| 模型名称 | 向量维度 |
|---------|---------|
| `nomic-embed-text` | 768 |
| `text-embedding-ada-002` | 1536 |
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |

#### 配置示例

```python
# 方式1：显式指定维度（推荐，适用于任意模型）
config = {
    'embedding_model': 'custom-model',
    'embedding_dim': 1024  # 显式指定维度
}

# 方式2：使用已知模型（自动推断）
config = {
    'embedding_model': 'nomic-embed-text'  # 自动推断为768维
}

# 方式3：运行时传入维度
result = graph_rag.query(
    "查询内容",
    data_path="/path/to/index",
    context={'embedding_dim': 768}
)
```

#### 配置合并规则

当同时存在多个配置来源时，按以下规则合并：

- **字典解包合并**：后面的字典覆盖前面的字典
- **配置优先级**：`context 参数 > 初始化 config 参数 > settings.yaml`

```python
# 示例：context 覆盖初始化配置
graph_rag = GraphRAG(config={'embedding_dim': 768})
result = graph_rag.query(
    "查询",
    context={'embedding_dim': 1024}  # 实际使用 1024
)
```

## 文件结构
- `graph_rag_impl.py` - 使用微软GraphRAG的实现
- `__init__.py` - 模块初始化文件
- `readme.md` - 说明文档