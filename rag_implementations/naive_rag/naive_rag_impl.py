"""
Naive RAG Implementation using LlamaIndex
基于LlamaIndex的简单RAG实现
"""
from typing import Dict, Any
from interfaces.rag_interface import RAGInterface
from config.config import settings
import logging


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
        self.index = None
        self.documents = []

        # 从配置中获取参数
        self.api_url = settings.naive_rag_api_url
        self.api_key = settings.naive_rag_api_key
        self.model = settings.naive_rag_model
        self.embedding_model = settings.naive_rag_embedding_model
        self.chunk_size = settings.naive_rag_chunk_size
        self.top_k = settings.naive_rag_top_k
        self.temperature = settings.naive_rag_temperature

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 尝试导入LlamaIndex，如果不存在则后续处理
        try:
            from llama_index.core import VectorStoreIndex, Document
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding

            self._llama_index_available = True
            self.VectorStoreIndex = VectorStoreIndex
            self.Document = Document
            self.OpenAI = OpenAI
            self.OpenAIEmbedding = OpenAIEmbedding

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

        try:
            # 从上下文获取文档，如果存在
            if context and 'documents' in context:
                documents_data = context['documents']
                if isinstance(documents_data, list):
                    # 将文档数据转换为LlamaIndex文档
                    llama_docs = []
                    for doc_data in documents_data:
                        if isinstance(doc_data, str):
                            llama_docs.append(self.Document(text=doc_data))
                        elif isinstance(doc_data, dict) and 'text' in doc_data:
                            llama_docs.append(self.Document(text=doc_data['text']))

                    # 创建索引 - 使用配置中的嵌入模型
                    self.index = self.VectorStoreIndex.from_documents(
                        llama_docs,
                        embed_model=self.OpenAIEmbedding(model=self.embedding_model)
                    )
                else:
                    # 如果没有提供文档，使用默认文档或示例数据
                    self.logger.info("未提供文档，使用示例数据")
                    llama_docs = [self.Document(text="这是一个示例文档，用于演示Naive RAG功能。")]
                    self.index = self.VectorStoreIndex.from_documents(
                        llama_docs,
                        embed_model=self.OpenAIEmbedding(model=self.embedding_model)
                    )
            else:
                # 如果没有上下文，使用示例数据
                llama_docs = [self.Document(text="这是一个示例文档，用于演示Naive RAG功能。")]
                self.index = self.VectorStoreIndex.from_documents(
                    llama_docs,
                    embed_model=self.OpenAIEmbedding(model=self.embedding_model)
                )

            # 执行查询 - 使用配置中的模型和API参数
            query_engine = self.index.as_query_engine(
                llm=self.OpenAI(model=self.model, api_key=self.api_key, api_base=self.api_url)
            )
            response = query_engine.query(query)

            return str(response)

        except Exception as e:
            self.logger.error(f"执行RAG查询时出错: {str(e)}")
            return f"错误：执行查询时出现问题 - {str(e)}"

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