"""
Naive RAG Implementation using LlamaIndex
基于LlamaIndex的简单RAG实现
"""
from typing import Dict, Any, List
from interfaces.rag_interface import RAGInterface
from config.config import settings
import logging
import time
import requests

import os


class VLLMEmbedding:
    """
    vLLM Embedding 类，支持 OpenAI 兼容的 embedding API
    
    用于绕过 LlamaIndex OpenAIEmbedding 的模型名限制，
    支持任意自定义 embedding 模型名称。
    
    注意：此类会在 _setup_global_embed_model 中被转换为 LlamaIndex 兼容的 embedding 对象。
    """
    
    def __init__(self, api_base: str, model: str, timeout: float = 60.0):
        """
        初始化 VLLMEmbedding
        
        Args:
            api_base: API 基础 URL（不含 /v1/embeddings）
            model: 模型名称
            timeout: 请求超时时间
        """
        self._api_base = api_base.rstrip('/')
        self._model = model
        self._timeout = timeout
    
    def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding"""
        return self._call_api(text)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """获取查询的 embedding"""
        return self._call_api(query)
    
    def _call_api(self, text: str) -> List[float]:
        """调用 vLLM embedding API"""
        response = requests.post(
            f'{self._api_base}/v1/embeddings',
            json={'input': text, 'model': self._model},
            timeout=self._timeout
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']


def _create_llama_index_embedding(api_base: str, model: str, timeout: float = 60.0):
    """
    创建 LlamaIndex 兼容的 embedding 对象
    
    使用自定义类继承 BaseEmbedding，绕过 OpenAIEmbedding 的模型名限制。
    """
    from llama_index.core.base.embeddings.base import BaseEmbedding
    
    class _VLLMEmbeddingAdapter(BaseEmbedding):
        """LlamaIndex BaseEmbedding 适配器"""
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


def _create_vllm_llm(api_base: str, model: str, temperature: float = 0.0, timeout: float = 300.0):
    """
    创建 LlamaIndex 兼容的 vLLM LLM 对象
    
    使用自定义类继承 CustomLLM，绕过 OpenAI 类的模型名限制。
    """
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    
    class _VLLMLLMAdapter(CustomLLM):
        """LlamaIndex CustomLLM 适配器，用于 vLLM"""
        
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
            """流式生成（简化实现，直接调用 complete）"""
            yield self.complete(prompt, **kwargs)
    
    return _VLLMLLMAdapter(api_base=api_base, model=model, temperature=temperature, timeout=timeout)

class NaiveRAG(RAGInterface):
    """
    基于LlamaIndex的简单RAG实现
    Simple RAG implementation using LlamaIndex
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化NaiveRAG

        Args:
            config: 配置参数字典
        """
        self.config = config or {}

        # 优先使用传入的配置，否则使用全局配置
        self.api_url = self.config.get('api_url', settings.naive_rag_api_url)
        self.api_key = self.config.get('api_key', settings.naive_rag_api_key)
        self.model = self.config.get('model', settings.naive_rag_model)
        self.embedding_model = self.config.get('embedding_model', settings.naive_rag_embedding_model)
        # embedding_url: 独立的 embedding 端点（默认使用 api_url）
        self.embedding_url = self.config.get('embedding_url', settings.naive_rag_embedding_url)
        # embedding_provider: 可选值 'ollama', 'openai', 'vllm', 或 'auto'（自动检测）
        self.embedding_provider = self.config.get('embedding_provider', settings.naive_rag_embedding_provider)
        self.chunk_size = self.config.get('chunk_size', settings.naive_rag_chunk_size)
        self.top_k = self.config.get('top_k', settings.naive_rag_top_k)
        self.temperature = self.config.get('temperature', settings.naive_rag_temperature)

        self.index = None
        self.documents = []
        self.is_index_initialized = False

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 记录最后一次检索的时间（用于性能评测）
        self.last_retrieval_time = 0.0

        # 记录最后一次生成的时间和token信息
        self.last_generation_time = 0.0
        self.last_generation_tokens = 0
        self.last_total_tokens = 0

        # 自动处理 Ollama API URL（移除 /v1 后缀）
        self.api_url = self._adjust_api_url_for_ollama()

        # 尝试导入LlamaIndex核心组件，如果不存在则后续execute的时候是不会执行的（会报错说没安装）
        try:
            from llama_index.core import VectorStoreIndex, Document, Settings
            from llama_index.llms.openai import OpenAI

            self._llama_index_available = True
            self.VectorStoreIndex = VectorStoreIndex
            self.Document = Document
            self.OpenAI = OpenAI

            self.logger.info("LlamaIndex 核心模块导入成功")

            # 设置全局嵌入模型（仅在初始化时设置一次）
            self._setup_global_embed_model()

        except ImportError as e:
            self._llama_index_available = False
            self.logger.error(f"LlamaIndex 导入失败: {str(e)}")
            self.logger.warning("LlamaIndex not available. Please install it using: pip install llama-index")
        except Exception as e:
            self._llama_index_available = False
            self.logger.error(f"LlamaIndex 初始化时发生未知错误: {str(e)}")
            self.logger.warning("LlamaIndex not available. Please install it using: pip install llama-index")

    def _adjust_api_url_for_ollama(self) -> str:
        """根据模型类型调整 API URL，如果是 Ollama 模型则移除 /v1 后缀"""
        # 如果模型是 Ollama 模型，且 URL 以 /v1 结尾，则移除它
        if self._is_ollama_model(self.model) or self._is_ollama_endpoint():
            # 移除末尾的 /v1 或 /v1/
            adjusted_url = self.api_url.rstrip('/')
            if adjusted_url.endswith('/v1'):
                adjusted_url = adjusted_url[:-3]  # 移除 /v1
                self.logger.info(f"检测到 Ollama 模型，调整 API URL 从 {self.api_url} 到 {adjusted_url}")
            return adjusted_url
        return self.api_url

    def _setup_global_embed_model(self):
        """设置全局嵌入模型
        
        支持三种 embedding 提供者:
        - ollama: 使用 OllamaEmbedding（Ollama 专用）
        - vllm: 使用 VLLMEmbedding（vLLM 专用，支持自定义模型名）
        - openai: 使用 OpenAIEmbedding（仅支持 OpenAI 官方模型）
        - auto: 自动检测（根据模型名和端点判断）
        """
        from llama_index.core import Settings

        # 确定使用哪种 embedding 提供者
        provider = self._detect_embedding_provider()
        
        if provider == 'ollama':
            # 使用 Ollama 嵌入模型
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
                embed_model = OllamaEmbedding(
                    model_name=self.embedding_model,
                    base_url=self.embedding_url,
                    ollama_additional_kwargs={"temperature": self.temperature},
                )
                self.logger.info(f"使用 Ollama embedding 模型: {self.embedding_model} @ {self.embedding_url}")
            except Exception as e:
                self.logger.error(f"无法初始化 Ollama embedding 模型 {self.embedding_model}: {str(e)}")
                self.logger.info("请确保 Ollama 服务正在运行，并且模型已下载。例如：ollama pull nomic-embed-text")
                raise
        elif provider == 'vllm':
            # 使用自定义 embedding（支持任意模型名，兼容 LlamaIndex）
            try:
                embed_model = _create_llama_index_embedding(
                    api_base=self.embedding_url,
                    model=self.embedding_model,
                    timeout=300.0
                )
                self.logger.info(f"使用 vLLM embedding 模型: {self.embedding_model} @ {self.embedding_url}")
            except Exception as e:
                self.logger.error(f"无法初始化 vLLM embedding 模型 {self.embedding_model}: {str(e)}")
                raise
        else:
            # provider == 'openai': 使用 OpenAI 官方 embedding 模型
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            # 构建正确的 API 端点 URL
            api_base = self._get_embedding_api_base()
            
            embed_model = OpenAIEmbedding(
                api_key=self.api_key,
                api_base=api_base,
                model=self.embedding_model,
                timeout=300.0  # 增加超时时间
            )
            self.logger.info(f"使用 OpenAI embedding 模型: {self.embedding_model} (provider: {provider})")

        # 设置到全局配置（关键步骤）
        Settings.embed_model = embed_model
    
    def _detect_embedding_provider(self) -> str:
        """检测 embedding 提供者类型
        
        Returns:
            str: 'ollama', 'openai', 或 'vllm'
        """
        # 如果显式指定了 provider，则使用指定的值
        if self.embedding_provider != 'auto':
            provider = self.embedding_provider.lower()
            if provider in ['ollama', 'openai', 'vllm']:
                return provider
        
        # 自动检测：检查是否是 Ollama 特有的 embedding 模型
        ollama_embedding_models = ['nomic-embed-text', 'mxbai-embed-large', 'all-minilm']
        if self.embedding_model in ollama_embedding_models:
            return 'ollama'
        
        # 检查端点类型
        if self._is_ollama_endpoint():
            return 'ollama'
        elif self._is_vllm_endpoint():
            return 'vllm'
        
        # 默认使用 OpenAI 兼容模式
        return 'openai'
    
    def _is_vllm_endpoint(self) -> bool:
        """判断是否为 vLLM 服务端点"""
        # 检查 embedding_url 或 api_url 是否包含 vLLM 特征
        url_to_check = self.embedding_url or self.api_url
        return '8000' in url_to_check or '8001' in url_to_check or 'vllm' in url_to_check.lower()
    
    def _get_embedding_api_base(self) -> str:
        """获取 embedding API 的基础 URL
        
        对于 vLLM 和 OpenAI 兼容服务，需要确保 URL 包含 /v1 后缀
        """
        api_base = self.api_url.rstrip('/')
        
        # 如果 URL 已经包含 /v1，直接返回
        if api_base.endswith('/v1'):
            return api_base
        
        # 对于 vLLM 和 OpenAI 兼容服务，添加 /v1 后缀
        if self._is_vllm_endpoint() or not self._is_ollama_endpoint():
            api_base = api_base + '/v1'
            self.logger.info(f"为 OpenAI 兼容 embedding 服务调整 URL: {api_base}")
        
        return api_base


    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        执行RAG查询
        Execute RAG query

        Args:
            query (str): 查询字符串
            context (Dict[str, Any], optional): 上下文信息

        Returns:
            str: 查询结果
        """
        if not self._llama_index_available:
            return f"错误：LlamaIndex库未安装。无法执行查询: {query}"

        if not self.is_index_initialized:
            # 如果没有预构建索引，检查是否在上下文中有文档数据
            if context and 'documents' in context:
                # 如果有文档数据，构建索引（向后兼容）
                documents_data = context['documents']
                if isinstance(documents_data, list):
                    # 将文档数据转换为LlamaIndex文档
                    llama_docs = []
                    for doc_data in documents_data:
                        if isinstance(doc_data, str):
                            llama_docs.append(self.Document(text=doc_data))
                        elif isinstance(doc_data, dict) and 'text' in doc_data:
                            llama_docs.append(self.Document(text=doc_data['text']))
                    self.build_index_from_data([doc.text for doc in llama_docs])
                else:
                    # 没有有效文档数据，返回错误
                    return "错误：没有可用的索引，也未提供文档数据用于构建索引"
            else:
                # 没有预构建索引，也未提供文档数据，返回错误
                return "错误：没有可用的索引。请先调用build_index方法构建索引，或在context中提供文档数据。"

        try:
            # 1. 先执行检索（记录检索时间）
            retriever = self.index.as_retriever(similarity_top_k=self.top_k)
            retrieval_start = time.time()
            nodes = retriever.retrieve(query)
            retrieval_end = time.time()
            self.last_retrieval_time = retrieval_end - retrieval_start

            # 2. 再执行生成 - 使用自定义 prompt 模板
            # 定义优化的 prompt 模板
            from llama_index.core import PromptTemplate

            # 获取配置中的 prompt 模板，如果没有则使用默认模板
            prompt_template_str = self.config.get('prompt_template',
"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.

Guidelines:
1. Provide ONLY the answer, no explanations or reasoning
2. Keep the answer as short as possible - typically 1-5 words
3. For yes/no questions, answer only "yes" or "no"
4. For dates, use the exact format (e.g., "December 31, 2015")
5. For numbers, provide just the number (e.g., "1522")
6. For names, provide just the name (e.g., "Terry Crews")
7. DO NOT include phrases like "The answer is", "According to", "Based on", etc.
8. DO NOT add any additional context or information
9. If the answer is not in the context, say "I don't know"

Query: {query_str}
Answer:""")

            # 创建 PromptTemplate
            prompt_template = PromptTemplate(prompt_template_str)

            # 检查是否使用 Ollama 模型，如果是则使用 Ollama LLM
            if self._is_ollama_model(self.model):
                from llama_index.llms.ollama import Ollama
                llm = Ollama(
                    model=self.model,
                    base_url=self.api_url,
                    temperature=self.temperature,
                    request_timeout=300.0
                )
            elif self._is_vllm_endpoint():
                # 使用自定义 vLLM LLM（支持任意模型名）
                llm = _create_vllm_llm(
                    api_base=self.api_url,
                    model=self.model,
                    temperature=self.temperature
                )
            else:
                # 使用 OpenAI 官方 LLM（仅支持 OpenAI 官方模型）
                llm = self.OpenAI(model=self.model, api_key=self.api_key, api_base=self.api_url)

            # 创建自定义的 query_engine
            query_engine = self.index.as_query_engine(
                llm=llm,
                text_qa_template=prompt_template,
                similarity_top_k=self.top_k
            )
            # print("执行查询...")

            generation_start = time.time()
            response = query_engine.query(query)
            generation_end = time.time()

            # 记录生成时间
            self.last_generation_time = generation_end - generation_start

            # 注意：LlamaIndex的query_engine可能不直接返回usage信息
            # 如果需要token信息，可能需要更底层的API调用
            self.last_generation_tokens = 0  # 暂时设为0，后续可以优化
            self.last_total_tokens = 0

            # print("response", str(response))
            return str(response)

        except Exception as e:
            self.logger.error(f"执行RAG查询时出错: {str(e)}")
            return f"错误：执行查询时出现问题 - {str(e)}"

    def _is_ollama_model(self, model_name: str) -> bool:
        """判断是否为 Ollama 模型"""
        # HuggingFace 格式的模型名（包含 /）不是 Ollama 模型
        if '/' in model_name:
            return False
        
        # 检查模型名称是否包含 Ollama 特有的参数格式（如 :3b, :7b 等）
        ollama_param_patterns = [':3b', ':7b', ':8b', ':13b', ':70b', ':9b', ':34b', ':67b', ':110b', ':latest']

        model_lower = model_name.lower()

        # 如果包含特定的参数标签（如 :3b），则几乎可以确定是 Ollama 模型
        if any(pattern in model_lower for pattern in ollama_param_patterns):
            return True

        # 检查是否包含常见的 Ollama 模型前缀
        ollama_model_prefixes = [
            'qwen', 'llama', 'mistral', 'mixtral', 'phi3', 'gemma', 'yi', 'codellama',
            'command-r', 'nomic', 'mxbai', 'all-'
        ]

        # 检查是否匹配 Ollama 模型前缀且包含参数
        for prefix in ollama_model_prefixes:
            if prefix in model_lower and (':' in model_name or any(c.isdigit() for c in model_name)):
                return True

        return False

    def _is_ollama_endpoint(self) -> bool:
        """判断是否为 Ollama 服务端点"""
        # 如果 API URL 包含 11434 端口，很可能是 Ollama 服务
        return '11434' in self.api_url or 'ollama' in self.api_url.lower()

    def build_index_from_data(self, data, metadata=None, show_progress=True, **kwargs):
        """
        从数据列表构建索引（适用于内存驱动的RAG）
        Build index from data list (suitable for in-memory RAG)

        Args:
            data (List[str]): 用于构建索引的文档数据列表
            metadata (Optional[List[Dict[str, Any]]]): 与文档关联的元数据列表
            show_progress (bool): 是否显示进度条（默认为True）
            **kwargs: 额外的参数，用于特定实现的配置

        Returns:
            bool: 构建成功返回True，失败返回False
        """
        if not self._llama_index_available:
            self.logger.error("LlamaIndex not available. Please install it using: pip install llama-index")
            return False

        try:
            # 确保嵌入模型设置正确
            self._setup_global_embed_model()

            # 处理输入数据
            llama_docs = []
            if metadata and len(metadata) == len(data):
                # 如果提供了元数据且与数据长度匹配，则配对使用
                for text, meta in zip(data, metadata):
                    llama_docs.append(self.Document(text=text, metadata=meta or {}))
            else:
                # 否则，只使用文本内容
                for text in data:
                    llama_docs.append(self.Document(text=text))

            # 创建索引 - 使用全局设置中的嵌入模型
            self.index = self.VectorStoreIndex.from_documents(
                llama_docs,
                show_progress=show_progress
            )

            # 标记索引已初始化
            self.is_index_initialized = True
            self.documents = llama_docs  # 保存文档列表用于后续操作

            self.logger.info(f"成功构建索引，包含 {len(llama_docs)} 个文档")
            return True

        except Exception as e:
            import traceback
            self.logger.error(f"构建索引时出错: {str(e)}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False

    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """
        添加文档到索引
        Add a document to the index

        Args:
            text (str): 文档文本
            metadata (Dict[str, Any], optional): 元数据
        """
        if not self._llama_index_available:
            self.logger.error("LlamaIndex未安装，无法添加文档")
            return

        from llama_index.core import Document
        doc = Document(text=text, metadata=metadata or {})
        self.documents.append(doc)

        # 如果索引已存在，需要重建索引
        if self.index is not None:
            self.index = self.VectorStoreIndex.from_documents(
                self.documents,
                embed_model=self.OpenAIEmbedding(model=self.embedding_model)
            )

    def save_index(self, storage_dir: str) -> bool:
        """
        保存索引到磁盘

        Args:
            storage_dir: 存储目录路径

        Returns:
            bool: 保存成功返回 True
        """
        if not self.is_index_initialized:
            self.logger.error("索引未初始化，无法保存")
            return False

        try:
            os.makedirs(storage_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=storage_dir)
            self.logger.info(f"索引已保存到: {storage_dir}")
            return True
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
            return False

    def load_index(self, storage_dir: str) -> bool:
        """
        从磁盘加载索引

        Args:
            storage_dir: 存储目录路径

        Returns:
            bool: 加载成功返回 True
        """
        try:
            from llama_index.core import StorageContext, load_index_from_storage

            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            self.index = load_index_from_storage(storage_context)

            # 重新设置嵌入模型（重要！）- 使用统一的方法
            self._setup_global_embed_model()

            self.is_index_initialized = True
            self.logger.info(f"索引已从 {storage_dir} 加载")
            return True
        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            return False