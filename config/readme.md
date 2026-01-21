# 配置系统 (Configuration System)

此模块定义每个组件的设置。

## 功能
- 为每个组件定义配置选项
- 允许在实现之间切换
- 提供配置验证和加载功能
- 支持 API 和本地 Ollama 模型切换

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

### 切换模型提供商

只需修改 `.env` 文件中的 `API_BASE_URL` 和 `settings.yaml` 中的模型名称即可在 API 和 Ollama 之间切换。

## 组件配置

每个组件（Decomposer、Router、NoRAG、NaiveRAG、GraphRAG）都有独立的配置部分，可以在 `settings.yaml` 中单独配置。