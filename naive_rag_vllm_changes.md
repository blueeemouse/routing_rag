# NaiveRAG vLLM 支持改动说明

## 改动概述

为 NaiveRAG 添加纯 vLLM 支持，允许 LLM 和 Embedding 使用独立的 vLLM 端点。

## 问题背景

1. LlamaIndex 的 `OpenAIEmbedding` 只支持 OpenAI 官方模型名，拒绝自定义模型名如 `nomic-ai/nomic-embed-text-v1`
2. LlamaIndex 的 `OpenAI` LLM 类同样只支持 OpenAI 官方模型名
3. Ollama 和 vLLM 使用不同的 API 端点和协议

## 解决方案

创建自定义适配器类，绕过 LlamaIndex 的模型名限制。

---

## 文件改动详情

### 1. config/config.py

**新增属性**：

```python
@property
def naive_rag_embedding_url(self) -> str:
    """获取naive_rag embedding URL（默认使用api_url）"""
    return self.config.get('naive_rag', {}).get('embedding_url', self.naive_rag_api_url)

@property
def naive_rag_embedding_provider(self) -> str:
    """获取naive_rag embedding provider（ollama/openai/vllm/auto）"""
    return self.config.get('naive_rag', {}).get('embedding_provider', 'auto')
```

---

### 2. config/settings.yaml

**新增 vLLM 配置示例**：

```yaml
# 选项 3: 使用本地 vLLM
# naive_rag:
#   api_url: "http://127.0.0.1:8000/v1"  # vLLM LLM 端点
#   api_key: "EMPTY"
#   model: "Qwen/Qwen2.5-3B-Instruct"  # HuggingFace 格式
#   embedding_model: "nomic-ai/nomic-embed-text-v1"
#   embedding_url: "http://127.0.0.1:8001"  # vLLM embedding 端点
#   embedding_provider: "vllm"
#   chunk_size: 512
#   top_k: 5
#   temperature: 0.0
```

---

### 3. rag_implementations/naive_rag/naive_rag_impl.py

#### 3.1 新增 Embedding 适配器函数

```python
def _create_llama_index_embedding(api_base: str, model: str, timeout: float = 60.0):
    """
    创建 LlamaIndex 兼容的 embedding 对象
    使用自定义类继承 BaseEmbedding，绕过 OpenAIEmbedding 的模型名限制。
    """
    from llama_index.core.base.embeddings.base import BaseEmbedding
    
    class _VLLMEmbeddingAdapter(BaseEmbedding):
        embed_batch_size: int = 10
        
        def __init__(self, api_base: str, model: str, timeout: float = 60.0, **kwargs):
            super().__init__(**kwargs)
            self._api_base = api_base.rstrip('/')
            self._model = model
            self._timeout = timeout
        
        def _get_query_embedding(self, query: str) -> List[float]:
            return self._call_api(query)
        
        def _get_text_embedding(self, text: str) -> List[float]:
            return self._call_api(text)
        
        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [self._call_api(t) for t in texts]
        
        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._call_api(query)
        
        def _call_api(self, text: str) -> List[float]:
            response = requests.post(
                f'{self._api_base}/v1/embeddings',
                json={'input': text, 'model': self._model},
                timeout=self._timeout
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
    
    return _VLLMEmbeddingAdapter(api_base=api_base, model=model, timeout=timeout)
```

#### 3.2 新增 LLM 适配器函数

```python
def _create_vllm_llm(api_base: str, model: str, temperature: float = 0.0, timeout: float = 300.0):
    """
    创建 LlamaIndex 兼容的 vLLM LLM 对象
    使用自定义类继承 CustomLLM，绕过 OpenAI 类的模型名限制。
    """
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    
    class _VLLMLLMAdapter(CustomLLM):
        def __init__(self, api_base: str, model: str, temperature: float = 0.0, timeout: float = 300.0, **kwargs):
            super().__init__(**kwargs)
            # 确保 URL 包含 /v1 前缀
            self._api_base = api_base.rstrip('/')
            if not self._api_base.endswith('/v1'):
                self._api_base = self._api_base + '/v1'
            self._model = model
            self._temperature = temperature
            self._timeout = timeout
        
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(model_name=self._model)
        
        def complete(self, prompt: str, **kwargs) -> CompletionResponse:
            response = requests.post(
                f'{self._api_base}/chat/completions',
                json={
                    'model': self._model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self._temperature,
                    'max_tokens': 512
                },
                headers={'Content-Type': 'application/json'},
                timeout=self._timeout
            )
            response.raise_for_status()
            return CompletionResponse(text=response.json()['choices'][0]['message']['content'])
        
        def stream_complete(self, prompt: str, **kwargs):
            yield self.complete(prompt, **kwargs)
    
    return _VLLMLLMAdapter(api_base=api_base, model=model, temperature=temperature, timeout=timeout)
```

#### 3.3 修改 __init__ 方法

新增配置项读取：

```python
# embedding_url: 独立的 embedding 端点（默认使用 api_url）
self.embedding_url = self.config.get('embedding_url', settings.naive_rag_embedding_url)
# embedding_provider: 可选值 'ollama', 'openai', 'vllm', 或 'auto'
self.embedding_provider = self.config.get('embedding_provider', settings.naive_rag_embedding_provider)
```

#### 3.4 修改 _setup_global_embed_model 方法

新增 vllm provider 分支：

```python
elif provider == 'vllm':
    embed_model = _create_llama_index_embedding(
        api_base=self.embedding_url,
        model=self.embedding_model,
        timeout=300.0
    )
```

#### 3.5 修改 execute 方法中的 LLM 选择逻辑

```python
if self._is_ollama_model(self.model):
    from llama_index.llms.ollama import Ollama
    llm = Ollama(...)
elif self._is_vllm_endpoint():
    llm = _create_vllm_llm(
        api_base=self.api_url,
        model=self.model,
        temperature=self.temperature
    )
else:
    llm = self.OpenAI(...)
```

#### 3.6 修改 _is_ollama_model 方法

HuggingFace 格式的模型名（包含 `/`）不应被识别为 Ollama 模型：

```python
def _is_ollama_model(self, model_name: str) -> bool:
    # HuggingFace 格式的模型名（包含 /）不是 Ollama 模型
    if '/' in model_name:
        return False
    # ... 其余逻辑不变
```

#### 3.7 修改 _is_vllm_endpoint 方法

检查 embedding_url：

```python
def _is_vllm_endpoint(self) -> bool:
    url_to_check = self.embedding_url or self.api_url
    return '8000' in url_to_check or '8001' in url_to_check or 'vllm' in url_to_check.lower()
```

---

## 配置示例

### Ollama 配置（一个端口）

```yaml
naive_rag:
  api_url: "http://127.0.0.1:11434"
  api_key: "ollama"
  model: "qwen2.5:3b"
  embedding_model: "nomic-embed-text"
  # embedding_url 和 embedding_provider 不需要配置
```

### vLLM 配置（两个端口）

```yaml
naive_rag:
  api_url: "http://127.0.0.1:8000/v1"      # LLM 端点
  api_key: "EMPTY"
  model: "Qwen/Qwen2.5-3B-Instruct"        # HuggingFace 格式
  embedding_url: "http://127.0.0.1:8001"   # Embedding 端点
  embedding_model: "nomic-ai/nomic-embed-text-v1"
  embedding_provider: "vllm"
```

---

## 测试结果

- ✅ 纯 Ollama：测试通过
- ✅ 纯 vLLM：测试通过（查询 "What is the capital of France?" 返回 "Paris"）

---

## 依赖

无新增依赖，使用已有的 `requests` 库。
