"""
路由器接口定义
Router Interface Definition

定义查询路由器的通用接口，确保可交换的实现
Defines the common interface for query routers, ensuring swappable implementations
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class RouterInterface(ABC):
    """
    查询路由器接口
    Query Router Interface
    """
    
    @abstractmethod
    def route(self, sub_query: str) -> str:
        """
        确定如何处理子查询
        Determine how to process the sub-query
        
        Args:
            sub_query (str): 子查询字符串
            
        Returns:
            str: 处理策略（如 'no_rag', 'naive_rag', 'graph_rag'）
        """
        pass