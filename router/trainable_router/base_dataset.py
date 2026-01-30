"""
可训练路由器数据集基类

定义所有数据集需要实现的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


class BaseRouterDataset(Dataset):
    """路由器数据集基类"""
    
    def __init__(self, config, tokenizer=None):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器
        """
        self.config = config
        self.tokenizer = tokenizer
        self.data = []
    
    @abstractmethod
    def load_data(self, path: str):
        """
        加载数据
        
        Args:
            path: 数据路径
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项字典
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        获取数据集大小
        
        Returns:
            数据集大小
        """
        pass
    
    def tokenize(self, query: str) -> Dict[str, torch.Tensor]:
        """
        分词
        
        Args:
            query: 查询字符串
            
        Returns:
            分词结果字典
        """
        if self.tokenizer is None:
            return {'input_ids': torch.tensor([], dtype=torch.long)}
        
        max_length = getattr(self.config.training, 'max_length', 512) if hasattr(self.config, 'training') else 512
        
        return self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
