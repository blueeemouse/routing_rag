# 查询分解器模块 (Query Decomposer Module)

此模块负责将复杂查询拆分为子查询。

## 功能
- 将复杂查询分解为子查询
- 支持多种分解策略
- 提供统一的接口供其他模块调用
- 支持 API 和本地 Ollama 模型

## 文件结构
- `decomposer.py` - 查询分解器的实现
- `__init__.py` - 模块初始化文件

## 配置

在 `config/settings.yaml` 中配置：

### 使用 API 服务

```yaml
decomposer:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${DECOMPOSER_API_KEY}"
  model: "gpt-5.1"
  prompt: "将复杂查询分解为简单子查询。输入：{query}\n输出子查询，每行一个：（如果判断不需要分解，则返回原查询）"
```

### 使用本地 Ollama

```yaml
decomposer:
  api_url: "${API_BASE_URL}/chat/completions"
  api_key: "${DECOMPOSER_API_KEY}"
  model: "qwen2.5:3b"
  prompt: "将复杂查询分解为简单子查询。输入：{query}\n输出子查询，每行一个：（如果判断不需要分解，则返回原查询）"
```

## 模型支持

- **API 模型**: gpt-5.1, gpt-4o, deepseek-v3 等
- **Ollama 模型**: qwen2.5:3b, llama3.1:8b 等

## 使用示例

```python
from decomposer.decomposer import Decomposer

decomposer = Decomposer()
sub_queries = decomposer.decompose("什么是人工智能？它有哪些应用？")
print(sub_queries)
# 输出: ["什么是人工智能？它有哪些应用？"]
```

## 测试

### 测试 API 模型
```bash
python tests/test_decomposer.py
```

### 测试 Ollama 模型
```bash
python tests/test_ollama_decomposer.py
```