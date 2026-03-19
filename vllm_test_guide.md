# vLLM 支持测试指南

## 1. 服务器端准备

### 1.1 安装 vLLM
```bash
pip install vllm
```

### 1.2 启动 vLLM 服务

**启动 Chat 模型（用于 Decomposer、Router、NoRAG、NaiveRAG）：**
```bash
# 方式一：直接启动（自动下载模型）
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000

# 方式二：指定 GPU
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000

# 方式三：使用本地模型路径
vllm serve /path/to/Qwen2.5-3B-Instruct --port 8000
```

**启动 Embedding 模型（用于 NaiveRAG embedding）：**
```bash
# 可选：单独部署 embedding 模型在不同端口
vllm serve BAAI/bge-small-en-v1.5 --port 8001

# 或使用其他 embedding 模型
vllm serve BAAI/bge-base-en-v1.5 --port 8001
```

### 1.3 验证 vLLM 服务
```bash
# 测试 Chat API
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# 测试 Embedding API
curl http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-small-en-v1.5",
    "input": "Hello world"
  }'
```

---

## 2. 配置切换

### 2.1 修改 settings.yaml

将以下组件的配置从 Ollama 切换到 vLLM：

```yaml
# Decomposer - vLLM 配置
decomposer:
  api_url: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-3B-Instruct"
  temperature: 0.0
  # prompt 保持不变...

# Router - vLLM 配置
router:
  api_url: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-3B-Instruct"
  temperature: 0.0
  # prompt 保持不变...

# NoRAG - vLLM 配置
no_rag:
  api_url: "http://127.0.0.1:8000/v1/chat/completions"
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-3B-Instruct"
  temperature: 0.0
  # prompt_template 保持不变...

# NaiveRAG - vLLM 配置
naive_rag:
  api_url: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-3B-Instruct"
  embedding_model: "BAAI/bge-small-en-v1.5"
  embedding_provider: "vllm"  # 明确指定 vLLM
  chunk_size: 512
  top_k: 5
  temperature: 0.0
  # prompt_template 保持不变...

# LLM Router - vLLM 配置
llm_router:
  api_url: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-3B-Instruct"
  temperature: 0.0
  max_tokens: 20
  mode: "zero_shot"
  # 其他配置保持不变...
```

### 2.2 GraphRAG 配置

使用 `graphrag_vllm_hotpotqa_template/graphrag_vllm_config.yml`：

1. 复制模板目录
2. 修改路径
3. 在代码中指定使用该配置文件

---

## 3. 运行测试

### 3.1 简单测试脚本

```python
# test_vllm.py
from decomposer.decomposer import Decomposer
from rag_implementations.no_rag.no_rag_impl import NoRAG

# 测试 Decomposer
print("Testing Decomposer with vLLM...")
decomposer = Decomposer()
result = decomposer.decompose("什么是人工智能？它有哪些应用？")
print(f"Decomposer result: {result}")

# 测试 NoRAG
print("\nTesting NoRAG with vLLM...")
no_rag = NoRAG()
answer = no_rag.execute("北京是中国的首都吗？")
print(f"NoRAG answer: {answer}")
```

### 3.2 完整流程测试

```bash
# 运行已有的测试脚本（确保 settings.yaml 已切换到 vLLM）
python test_retrieval_strategies.py
```

---

## 4. 常见问题

### Q1: vLLM 启动失败 - 显存不足
```bash
# 减少显存占用
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 --gpu-memory-utilization 0.8
```

### Q2: Embedding API 不可用
vLLM 的 embedding 功能需要特定模型支持。如果不可用，可以：
- 使用 Ollama 的 embedding 模型（保持 `embedding_provider: ollama`）
- 或使用 OpenAI 的 embedding API

### Q3: 模型名称不匹配
确保 `settings.yaml` 中的 `model` 名称与 vLLM 启动时加载的模型名称完全一致。

### Q4: 端口冲突
```bash
# 使用其他端口
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8001
# 同时修改 settings.yaml 中的 api_url
```

---

## 5. 性能对比

测试时可以记录以下指标对比 Ollama 和 vLLM：

| 指标 | Ollama | vLLM |
|------|--------|------|
| 首 token 延迟 | ? | ? |
| 吞吐量 | ? | ? |
| 显存占用 | ? | ? |
| 并发能力 | ? | ? |
