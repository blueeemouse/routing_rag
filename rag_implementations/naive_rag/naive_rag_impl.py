"""
Naive RAG Implementation using LlamaIndex
基于LlamaIndex的简单RAG实现
"""
from typing import Dict, Any
from interfaces.rag_interface import RAGInterface
from config.config import settings
import logging
import time

import os

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

        # 尝试导入LlamaIndex，如果不存在则后续execute的时候是不会执行的（会报错说没安装）
        try:
            from llama_index.core import VectorStoreIndex, Document, Settings
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding

            self._llama_index_available = True
            self.VectorStoreIndex = VectorStoreIndex
            self.Document = Document
            self.OpenAI = OpenAI
            self.OpenAIEmbedding = OpenAIEmbedding

            # 在初始化函数里就设置好embedding model，以便加载索引时使用
            embed_model = OpenAIEmbedding(
                api_key=self.api_key,      # ← 统一使用 self.api_key
                api_base=self.api_url,     # ← 统一使用 self.api_url  
                model=self.embedding_model # ← 统一使用 self.embedding_model
            )

            # 2. 设置到全局配置（关键步骤）
            Settings.embed_model = embed_model

        except ImportError:
            self._llama_index_available = False
            self.logger.warning("LlamaIndex not available. Please install it using: pip install llama-index")
        
        
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

            # 2. 再执行生成
            query_engine = self.index.as_query_engine(
                llm=self.OpenAI(model=self.model, api_key=self.api_key, api_base=self.api_url)
            )
            print("执行查询...")

            generation_start = time.time()
            response = query_engine.query(query)
            generation_end = time.time()

            # 记录生成时间
            self.last_generation_time = generation_end - generation_start

            # 注意：LlamaIndex的query_engine可能不直接返回usage信息
            # 如果需要token信息，可能需要更底层的API调用
            self.last_generation_tokens = 0  # 暂时设为0，后续可以优化
            self.last_total_tokens = 0

            print("response", str(response))
            return str(response)

        except Exception as e:
            self.logger.error(f"执行RAG查询时出错: {str(e)}")
            return f"错误：执行查询时出现问题 - {str(e)}"

    def build_index_from_data(self, data, metadata=None, **kwargs):
        """
        从数据列表构建索引（适用于内存驱动的RAG）
        Build index from data list (suitable for in-memory RAG)

        Args:
            data (List[str]): 用于构建索引的文档数据列表
            metadata (Optional[List[Dict[str, Any]]]): 与文档关联的元数据列表
            **kwargs: 额外的参数，用于特定实现的配置

        Returns:
            bool: 构建成功返回True，失败返回False
        """
        if not self._llama_index_available:
            self.logger.error("LlamaIndex not available. Please install it using: pip install llama-index")
            return False

        try:
            
            #############################
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core import Settings, VectorStoreIndex
            
            embed_model = OpenAIEmbedding(
                api_key=self.api_key,      # ← 统一使用 self.api_key
                api_base=self.api_url,     # ← 统一使用 self.api_url  
                model=self.embedding_model # ← 统一使用 self.embedding_model
            )

            # 2. 设置到全局配置（关键步骤）
            Settings.embed_model = embed_model
            #############################
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
                llama_docs
            )

            # 标记索引已初始化
            self.is_index_initialized = True
            self.documents = llama_docs  # 保存文档列表用于后续操作

            self.logger.info(f"成功构建索引，包含 {len(llama_docs)} 个文档")
            return True

        except Exception as e:
            self.logger.error(f"构建索引时出错: {str(e)}")
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