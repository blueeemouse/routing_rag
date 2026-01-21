# No RAG 实现

此目录包含No RAG实现，它直接使用LLM进行响应，而无需检索增强。

## 组件说明

- `no_rag_impl.py`: No RAG的主要实现文件
  - 实现了RAGInterface接口
  - 直接调用LLM模型对查询进行回答
  - 适用于无需检索增强的简单查询场景
  - 支持 API 和本地 Ollama 模型

## 用途

NoRAG实现用于：
- 直接LLM响应策略
- 无需检索增强的查询
- 通用问答场景

## 配置

在 `config/settings.yaml` 中配置：

### 使用 API 服务

```yaml
no_rag:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${NAIVE_RAG_API_KEY}"
  model: "gpt-4o"
  temperature: 0.0
```

### 使用本地 Ollama

```yaml
no_rag:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${NAIVE_RAG_API_KEY}"
  model: "qwen2.5:3b"
  temperature: 0.0
```

## 模型支持

- **API 模型**: gpt-4o, gpt-3.5-turbo, deepseek-v3 等
- **Ollama 模型**: qwen2.5:3b, llama3.1:8b, codegeex4:latest 等

## 使用示例

```python
from rag_implementations.no_rag.no_rag_impl import NoRAG

# 使用默认配置
no_rag = NoRAG()
result = no_rag.execute("什么是人工智能？")
print(result)

# 使用自定义配置
config = {
    'api_url': 'http://127.0.0.1:11434/v1/chat/completions',
    'api_key': 'ollama',
    'model': 'qwen2.5:3b'
}
no_rag = NoRAG(config)
result = no_rag.execute("1+1等于几？")
print(result)
```

## 测试

### 测试 API 模型
```bash
python tests/test_no_rag.py
```

### 测试 Ollama 模型
```bash
python tests/test_ollama_norag.py
```

## 接口兼容性

`NoRAG` 实现了 `RAGInterface`，确保与orchestrator和其他组件的兼容性。