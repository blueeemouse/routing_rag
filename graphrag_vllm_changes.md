# GraphRAG vLLM 支持修改说明

## 修改概述

为支持纯 vLLM 模式运行 GraphRAG，对以下文件进行了修改。

---

## 1. 配置文件修改

### 文件：`graphrag_ollama_hotpotqa_15_test_data/graphrag_hotpotqa_config.yml`

**修改原因**：将 embedding 端点从 Ollama 切换到 vLLM，实现纯 vLLM 模式，避免 GPU 显存冲突。

**修改前**：
```yaml
default_embedding_model:
  # Ollama embedding 配置（vLLM embedding 需要单独部署）
  api_base: http://127.0.0.1:11434/v1
  api_key: ollama
  concurrent_requests: 25
  model: nomic-embed-text
  model_provider: openai
  request_timeout: 600.0
  type: embedding
```

**修改后**：
```yaml
default_embedding_model:
  # ===== vLLM embedding 配置（当前启用） =====
  # 修改原因：测试纯 vLLM 模式，避免与 Ollama 的 GPU 显存冲突
  # 对应：LLM 使用 vLLM @ 8000，Embedding 使用 vLLM @ 8001
  api_base: http://127.0.0.1:8001/v1
  api_key: EMPTY
  concurrent_requests: 25
  model: nomic-ai/nomic-embed-text-v1
  model_provider: openai
  request_timeout: 600.0
  type: embedding
  # ===== Ollama embedding 配置（已注释） =====
  # 原配置：使用 Ollama 的 nomic-embed-text 模型
  # api_base: http://127.0.0.1:11434/v1
  # api_key: ollama
  # model: nomic-embed-text
```

---

## 2. GraphRAG 实现代码修改

### 文件：`rag_implementations/graph_rag/graph_rag_impl.py`

**修改位置**：`_get_vector_store_schema` 方法中的 `dimension_map`

**修改原因**：支持 vLLM/HuggingFace 格式的 embedding 模型名，使 GraphRAG 能够识别 vLLM 部署的 embedding 模型维度。

**修改前**：
```python
# 根据模型推断维度
dimension_map = {
    'nomic-embed-text': 768,
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
}
```

**修改后**：
```python
# 根据模型推断维度
# 支持 Ollama 格式 (如 nomic-embed-text) 和 vLLM/HuggingFace 格式 (如 nomic-ai/nomic-embed-text-v1)
dimension_map = {
    # Ollama 格式
    'nomic-embed-text': 768,
    'mxbai-embed-large': 1024,
    'all-minilm': 384,
    # vLLM/HuggingFace 格式
    'nomic-ai/nomic-embed-text-v1': 768,
    'BAAI/bge-small-en-v1.5': 384,
    'BAAI/bge-base-en-v1.5': 768,
    'BAAI/bge-large-en-v1.5': 1024,
    # OpenAI 格式
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
}
```

---

## 测试结果

| 组件 | 纯 Ollama | 纯 vLLM |
|-----|----------|--------|
| GraphRAG | ✅ 通过 | ✅ 通过 |

---

## 纯 Ollama 测试

**配置修改**：将 LLM 改为 Ollama 已有的 `llama3:8b-instruct-fp16` 模型

```yaml
default_chat_model:
  api_base: http://127.0.0.1:11434/v1
  api_key: ollama
  model: llama3:8b-instruct-fp16
  model_provider: openai
```

**测试结果**：返回 "Paris" ✅ (耗时 6.61s)

---

## 纯 vLLM 测试
```bash
# 启动 vLLM LLM (端口 8000)
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 --gpu-memory-utilization 0.8

# 启动 vLLM Embedding (端口 8001)
vllm serve nomic-ai/nomic-embed-text-v1 --port 8001 --task embed --trust-remote-code

# 测试查询
export API_BASE_URL="http://127.0.0.1:8000/v1"
export GRAPHRAG_API_KEY="EMPTY"
python3 -c "
from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
graph_rag = GraphRAG({})
result = graph_rag.execute('What is the capital of France?', context={
    'data_path': 'graphrag_ollama_hotpotqa_15_test_data',
    'search_mode': 'local',
    'config_filename': 'graphrag_hotpotqa_config.yml'
})
print(result)
"
```

**测试结果**：返回 "Paris" ✅
