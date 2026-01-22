# 查询路由器模块 (Query Router Module)

此模块确定如何处理每个子查询（无RAG、朴素RAG或图RAG）。

## 功能
- 根据子查询类型决定处理策略
- 支持多种RAG处理方式：无RAG、朴素RAG、图RAG
- 提供统一的接口供其他模块调用
- 支持 API 和本地 Ollama 模型

## 文件结构
- `router.py` - 查询路由器的实现
- `__init__.py` - 模块初始化文件

## 配置

在 `config/settings.yaml` 中配置：

### 使用 API 服务

```yaml
router:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${ROUTER_API_KEY}"
  model: "deepseek-v3"
  prompt: "确定查询处理策略：no_rag, naive_rag, 或 graph_rag。查询：{sub_query}\n策略："
```

### 使用本地 Ollama

```yaml
router:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${ROUTER_API_KEY}"
  model: "qwen2.5:3b"
  temperature: 0.1
  prompt: |
    任务：确定查询处理策略

    可选策略：
    - no_rag: 直接回答，无需检索
    - naive_rag: 使用向量检索
    - graph_rag: 使用图检索

    规则：
    1. 简单事实性问题 → no_rag
    2. 需要信息检索的问题 → naive_rag
    3. 复杂关系查询 → graph_rag
    4. 只输出策略名称，不要任何解释

    查询：{sub_query}
    策略：
```

## 路由策略

- **no_rag**: 简单事实性问题，无需检索
- **naive_rag**: 需要信息检索的问题
- **graph_rag**: 复杂关系查询

## 模型支持

- **API 模型**: gpt-4o, gpt-3.5-turbo, deepseek-v3 等
- **Ollama 模型**: qwen2.5:3b, llama3.1:8b 等

## 使用示例

```python
from router.router import Router

router = Router()
strategy = router.route("什么是人工智能？")
print(strategy)  # 输出: no_rag, naive_rag, 或 graph_rag
```

## 测试

### 测试 API 模型
```bash
python tests/test_router.py
```

### 测试 Ollama 模型
```bash
python tests/test_ollama_router.py
```