# vLLM 本地模型支持 - 完整总结

## 测试结果

| 组件 | 纯 Ollama | 纯 vLLM | 修改需求 |
|------|-----------|---------|----------|
| No RAG | ✅ 原生支持 | ✅ 原生支持 | 无需修改 |
| NaiveRAG | ✅ 原生支持 | ✅ 需修改 | 自定义适配器 |
| GraphRAG | ✅ 原生支持 | ✅ 需修改 | 扩展 dimension_map |

## 涉及的库/依赖

### 1. vLLM (版本 0.17.1)
- **用途**: 本地 LLM 推理服务，提供 OpenAI 兼容 API
- **安装**: `pip install vllm==0.17.1` 或使用 wheel 文件
- **启动命令**:
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --gpu-memory-utilization 0.6
  ```

### 2. LlamaIndex (NaiveRAG 依赖)
- **问题**: `OpenAIEmbedding` 和 `OpenAI` LLM 类限制模型名为 OpenAI 官方模型
- **解决**: 自定义 `VLLMEmbedding` 和 `VLLMLLM` 适配器类

### 3. openai (Python SDK)
- **版本冲突**: vLLM 0.10.0 需要 openai<=1.90.0，fnllm (GraphRAG依赖) 需要 openai>=1.99.9
- **解决**: 升级 vLLM 到 0.17.1 (支持 openai>=1.99.1, <2.25.0)

### 4. Ollama
- **用途**: 本地 LLM 和 Embedding 模型服务
- **模型格式**: 简单名称如 `qwen2.5:3b`, `nomic-embed-text`
- **API 端点**: `http://127.0.0.1:11434/v1`

## 关键代码修改

### NaiveRAG (`rag_implementations/naive_rag/naive_rag_impl.py`)

1. **新增 VLLMEmbedding 类** (继承 `BaseEmbedding`)
   - 绕过 LlamaIndex 的模型名验证
   - 直接调用 vLLM embedding API

2. **新增 VLLMLLM 类** (继承 `LLM`)
   - 绕过 LlamaIndex 的 LLM 模型名验证
   - 支持流式和非流式输出

3. **修改 `_setup_global_embed_model()`**
   - 新增 `vllm` provider 分支

4. **修改 `execute()` 方法**
   - 支持 vLLM LLM 端点检测和调用

### GraphRAG (`rag_implementations/graph_rag/graph_rag_impl.py`)

1. **扩展 dimension_map**
   ```python
   dimension_map = {
       # Ollama 格式
       'nomic-embed-text': 768,
       'mxbai-embed-large': 1024,
       # vLLM/HuggingFace 格式
       'nomic-ai/nomic-embed-text-v1': 768,
       'BAAI/bge-small-en-v1.5': 384,
       'BAAI/bge-base-en-v1.5': 768,
       'BAAI/bge-large-en-v1.5': 1024,
   }
   ```

### 配置文件 (`config/config.py`, `config/settings.yaml`)

1. **新增 `naive_rag_embedding_url` 属性**
   - 支持独立的 embedding 服务端点

2. **新增 `naive_rag_embedding_provider` 属性**
   - 支持 `auto`, `ollama`, `vllm` 三种模式

## 配置切换指南

### No RAG
```python
# Ollama
NoRAG(config={
    'api_url': 'http://127.0.0.1:11434/v1',
    'api_key': 'ollama',
    'model': 'qwen2.5:3b'
})

# vLLM
NoRAG(config={
    'api_url': 'http://127.0.0.1:8000/v1',
    'api_key': 'EMPTY',
    'model': 'Qwen/Qwen2.5-3B-Instruct'
})
```

### NaiveRAG
```yaml
# settings.yaml - Ollama
naive_rag:
  api_url: http://127.0.0.1:11434/v1
  api_key: ollama
  model: qwen2.5:3b
  embedding_url: http://127.0.0.1:11434/v1
  embedding_model: nomic-embed-text
  embedding_provider: ollama

# settings.yaml - vLLM
naive_rag:
  api_url: http://127.0.0.1:8000/v1
  api_key: EMPTY
  model: Qwen/Qwen2.5-3B-Instruct
  embedding_url: http://127.0.0.1:8001/v1  # 可独立部署
  embedding_model: nomic-ai/nomic-embed-text-v1
  embedding_provider: vllm
```

### GraphRAG
```yaml
# graphrag_hotpotqa_config.yml - Ollama
models:
  default_chat_model:
    api_base: http://127.0.0.1:11434/v1
    api_key: ollama
    model: qwen2.5:3b
  default_embedding_model:
    api_base: http://127.0.0.1:11434/v1
    api_key: ollama
    model: nomic-embed-text

# graphrag_hotpotqa_config.yml - vLLM
models:
  default_chat_model:
    api_base: http://127.0.0.1:8000/v1
    api_key: EMPTY
    model: Qwen/Qwen2.5-3B-Instruct
  default_embedding_model:
    api_base: http://127.0.0.1:8001/v1
    api_key: EMPTY
    model: nomic-ai/nomic-embed-text-v1
```

## 模型名称对照

| 用途 | Ollama 格式 | vLLM/HuggingFace 格式 |
|------|-------------|----------------------|
| LLM | `qwen2.5:3b` | `Qwen/Qwen2.5-3B-Instruct` |
| LLM | `llama3:8b-instruct-fp16` | `meta-llama/Llama-3-8B-Instruct` |
| Embedding | `nomic-embed-text` | `nomic-ai/nomic-embed-text-v1` |
| Embedding | `mxbai-embed-large` | `mixedbread-ai/mxbai-embed-large-v1` |

## 测试记录

- 日期: 2026-03-20
- 测试模型: Qwen2.5-3B-Instruct
- 测试查询: "What is the capital of France?"
- 预期答案: Paris

| 组件 | Ollama | vLLM |
|------|--------|------|
| No RAG | ✅ Paris | ✅ Paris |
| NaiveRAG | ✅ (之前测试) | ✅ (之前测试) |
| GraphRAG | ✅ Paris | ✅ (之前测试) |
