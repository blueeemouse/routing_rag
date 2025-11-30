"""
RAG接口定义
RAG Interface Definition

定义RAG实现的通用接口，确保可交换的实现
Defines the common interface for RAG implementations, ensuring swappable implementations
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


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