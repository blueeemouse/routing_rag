"""
RAG接口定义
RAG Interface Definition

定义RAG实现的通用接口，确保可交换的实现
Defines the common interface for RAG implementations, ensuring swappable implementations
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class RAGInterface(ABC):
    """
    RAG实现接口
    RAG Implementation Interface
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def build_index(self, data: List[str], metadata: Optional[List[Dict[str, Any]]] = None, **kwargs) -> bool:
        """
        构建索引
        Build index from data

        Args:
            data (List[str]): 用于构建索引的文档数据列表
            metadata (Optional[List[Dict[str, Any]]]): 与文档关联的元数据列表
            **kwargs: 额外的参数，用于特定实现的配置

        Returns:
            bool: 构建成功返回True，失败返回False
        """
        pass