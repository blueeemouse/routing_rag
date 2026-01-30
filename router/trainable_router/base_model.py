"""
可训练路由器模型基类

定义所有路由器模型需要实现的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn


class BaseRouterModel(nn.Module, ABC):
    """路由器模型基类"""
    
    def __init__(self, config):
        """
        初始化
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if config.device != 'auto' else 'cpu')
    
    @abstractmethod
    def encode(self, queries: List[str]) -> torch.Tensor:
        """
        编码query列表为embedding
        
        Args:
            queries: query字符串列表
            
        Returns:
            shape: (batch_size, hidden_size)
        """
        pass
    
    @abstractmethod
    def get_strategy_embeddings(self) -> torch.Tensor:
        """
        获取策略embedding
        
        Returns:
            shape: (num_strategies, hidden_size)
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, query_emb: torch.Tensor, strategy_embs: torch.Tensor) -> torch.Tensor:
        """
        计算query和策略embedding之间的相似度
        
        Args:
            query_emb: query embedding, shape: (batch_size, hidden_size)
            strategy_embs: 策略embeddings, shape: (num_strategies, hidden_size)
            
        Returns:
            shape: (batch_size, num_strategies)
        """
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            
        Returns:
            query embeddings, shape: (batch_size, hidden_size)
        """
        pass
    
    @abstractmethod
    def route(self, queries: List[str]) -> List[str]:
        """
        路由决策：根据query选择最佳策略
        
        Args:
            queries: query字符串列表
            
        Returns:
            策略名称列表
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        pass
