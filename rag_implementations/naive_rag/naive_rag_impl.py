"""
Naive RAG Implementation using LlamaIndex
基于LlamaIndex的简单RAG实现
"""
from typing import Dict, Any
from interfaces.rag_interface import RAGInterface
from config.config import settings
import logging

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
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core import Settings, VectorStoreIndex
            # 1. 创建自定义配置的嵌入模型
            embed_model = OpenAIEmbedding(
                api_key=os.getenv('NAIVE_RAG_API_KEY', os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE')),
                api_base="https://api.agicto.cn/v1",  # 替换为你的 base URL
                model="text-embedding-ada-002"  # 或你的服务商支持的模型
            )

            # 2. 设置到全局配置（关键步骤）
            Settings.embed_model = embed_model
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
                # llama_docs = [self.Document(text="人工智能（AI）是计算机科学的分支，旨在模拟人类智能，让机器完成原本需要人类思考的任务。AI 主要分为弱人工智能（专注单一任务，如语音助手）和强人工智能（具备通用智能，尚未实现）两大类型。核心分支包括机器学习（让机器从数据中学习规律）、深度学习（基于神经网络的机器学习子集，如ChatGPT、图像识别）、计算机视觉（让机器“看懂”图像/视频，如人脸识别）、自然语言处理（让机器理解人类语言，如翻译、对话机器人）。AI 的典型应用有智能客服、自动驾驶、医疗影像分析、推荐系统等，已广泛应用于电商、交通、医疗等领域。")]
                llama_docs = [self.Document(text="这是一段测试内容")]
                # print("使用示例数据创建索引")
                

                

                

                # 3. 现在调用 from_documents 会自动使用你的配置
                self.index = self.VectorStoreIndex.from_documents(
                    llama_docs
                )
                print("没有提供上下文，所以使用示例数据创建索引")

            # 执行查询 - 使用配置中的模型和API参数

            query_engine = self.index.as_query_engine(
                llm=self.OpenAI(model=self.model, api_key=self.api_key, api_base=self.api_url)
            )
            print("执行查询...")
            response = query_engine.query(query)
            print("response", str(response))
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