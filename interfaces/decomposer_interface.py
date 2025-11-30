"""
分解器接口定义
Decomposer Interface Definition

定义查询分解器的通用接口，确保可交换的实现
Defines the common interface for query decomposers, ensuring swappable implementations
"""
from abc import ABC, abstractmethod
from typing import List


class DecomposerInterface(ABC):
    """
    查询分解器接口
    Query Decomposer Interface
    """
    
    @abstractmethod
    def decompose(self, query: str) -> List[str]:
        """
        将复杂查询分解为子查询
        Split complex query into sub-queries
        
        Args:
            query (str): 复杂查询字符串
            
        Returns:
            List[str]: 子查询列表
        """
        pass