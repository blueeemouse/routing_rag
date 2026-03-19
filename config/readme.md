# 配置系统 (Configuration System)

此模块定义每个组件的设置。

## 功能
- 为每个组件定义配置选项
- 允许在实现之间切换
- 提供配置验证和加载功能
- 支持 API、Ollama 和 vLLM 模型切换

## 文件结构
- `config.py` - 配置定义和管理
- `settings.yaml` - 默认配置文件
- `readme.md` - 说明文档

## 模型提供商配置

### 使用 API 服务（默认）

在 `.env` 文件中配置：
```env
API_BASE_URL=https://api.openai.com/v1
DECOMPOSER_API_KEY=your_api_key
ROUTER_API_KEY=your_api_key
NAIVE_RAG_API_KEY=your_api_key
GRAPHRAG_API_KEY=your_api_key
```

在 `settings.yaml` 中配置模型：
```yaml
decomposer:
  model: "gpt-4o"
  ...
```

### 使用本地 Ollama

在 `.env` 文件中配置：
```env
API_BASE_URL=http://127.0.0.1:11434/v1
DECOMPOSER_API_KEY=ollama
ROUTER_API_KEY=ollama
NAIVE_RAG_API_KEY=ollama
GRAPHRAG_API_KEY=ollama
```

在 `settings.yaml` 中配置模型：
```yaml
decomposer:
  model: "qwen2.5:3b"
  ...
```

### 使用本地 vLLM

vLLM 是高性能的 LLM 推理引擎，支持 OpenAI 兼容 API。

**启动 vLLM 服务：**
```bash
# 安装 vLLM
pip install vllm

# 启动 Chat 模型服务
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000

# 启动 Embedding 模型服务（可选，可单独部署）
vllm serve BAAI/bge-small-en-v1.5 --port 8001
```

**配置 vLLM 后端：**

在 `settings.yaml` 中配置：
```yaml
decomposer:
  api_url: "http://127.0.0.1:8000/v1"  # vLLM 端点
  api_key: "EMPTY"  # vLLM 不需要 API key
  model: "Qwen/Qwen2.5-3B-Instruct"  # HuggingFace 格式模型名

naive_rag:
  api_url: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-3B-Instruct"
  embedding_model: "BAAI/bge-small-en-v1.5"
  embedding_provider: "vllm"  # 明确指定 embedding 提供者
```

**vLLM 与 Ollama 的主要区别：**

| 特性 | Ollama | vLLM |
|------|--------|------|
| 默认端口 | 11434 | 8000 |
| 模型名格式 | qwen2.5:3b | Qwen/Qwen2.5-3B-Instruct |
| API Key | ollama | EMPTY（任意值） |
| URL 后缀 | 不需要 /v1 | 需要 /v1 |

### 切换模型提供商

只需修改配置文件中的 `api_url` 和 `model` 参数即可在 API、Ollama 和 vLLM 之间切换。

## 组件配置

每个组件（Decomposer、Router、NoRAG、NaiveRAG、GraphRAG）都有独立的配置部分，可以在 `settings.yaml` 中单独配置。

### NaiveRAG Embedding 配置

NaiveRAG 支持多种 embedding 提供者：

```yaml
naive_rag:
  # embedding_provider 可选值:
  # - 'ollama': 使用 OllamaEmbedding（Ollama 专用）
  # - 'vllm': 使用 OpenAIEmbedding 连接 vLLM
  # - 'openai': 使用 OpenAIEmbedding 连接 OpenAI API
  # - 'auto': 自动检测（默认）
  embedding_provider: "auto"
```

## GraphRAG 配置

GraphRAG 使用独立的 YAML 配置文件（如 `graphrag_hotpotqa_config.yml`）。项目提供了两个模板：

- `graphrag_ollama_hotpotqa_15_test_data/` - Ollama 后端配置
- `graphrag_vllm_hotpotqa_template/` - vLLM 后端配置模板

切换 GraphRAG 后端只需修改配置文件中的 `models` 部分：

```yaml
models:
  default_chat_model:
    api_base: http://127.0.0.1:8000/v1  # Ollama: 11434, vLLM: 8000
    model: Qwen/Qwen2.5-3B-Instruct     # 对应的模型名
    model_provider: openai              # 保持 openai 以使用兼容协议
```