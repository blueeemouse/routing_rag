# 朴素RAG实现 (Naive RAG Implementation)

此模块实现朴素RAG策略，基于LlamaIndex框架。

## 功能

- 基于向量检索的RAG实现
- 简单的检索-生成流程
- 提供统一的RAG接口
- 支持通过API配置LLM和嵌入模型
- 支持Ollama本地模型部署

## Ollama 本地部署支持

本模块支持使用 Ollama 在本地运行模型，无需依赖外部 API。

### 配置 Ollama

1. 安装并启动 Ollama：

   ```bash
   # 启动 Ollama 服务
   ollama serve
   ```
2. 下载所需模型（示例）：

   ```bash
   # 下载主模型
   ollama pull qwen2.5:3b

   # 下载嵌入模型
   ollama pull nomic-embed-text
   ```
3. 配置 settings.yaml：

   ```yaml
   naive_rag:
     api_url: "http://127.0.0.1:11434"  # Ollama 不需要 /v1 后缀
     api_key: "ollama"  # Ollama 通常使用 "ollama" 作为 API key
     model: "qwen2.5:3b"
     embedding_model: "nomic-embed-text"
     # ... 其他配置
   ```

### 模型命名约定

Ollama 模型通常使用以下格式：

- `模型名:参数量` (如 `qwen2.5:3b`, `llama3:8b`)
- 或者 `模型名:tag` (如 `mistral:latest`)

系统会自动检测这些模型并使用 Ollama 接口。

## 文件结构

- `naive_rag_impl.py` - 使用LlamaIndex的朴素RAG实现
- `__init__.py` - 模块初始化文件
- `readme.md` - 说明文档
