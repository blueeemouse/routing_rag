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

    # 适用于内存数据的RAG
    def build_index_from_data(self, data: List[str], metadata: Optional[List[Dict[str, Any]]] = None, **kwargs) -> bool:
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
        raise NotImplementedError("This RAG implementation does not support building index from data.")

    # 适用于文件系统驱动的RAG
    def build_index_from_path(self, root_dir: str, config_filepath: str = None, output_dir: str = None, **kwargs) -> bool:
        """
        从路径构建索引（适用于文件系统驱动的RAG）
        Build index from file path (suitable for file system-based RAG)

        Args:
            root_dir (str): 项目根目录路径，配置文件中的相对路径将相对于此目录解析
            config_filepath (str, optional): 配置文件路径
            output_dir (str, optional): 输出目录路径，可覆盖配置文件中的设置
            **kwargs: 额外的参数，用于特定实现的配置

        Returns:
            bool: 构建成功返回True，失败返回False
        """
        raise NotImplementedError("This RAG implementation does not support building index from a file path.")