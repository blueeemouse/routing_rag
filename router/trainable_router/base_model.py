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
        
        # 处理设备配置
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
    
    # ========== 必须实现的抽象方法 ==========
    
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
    
    # ========== 可选方法（提供默认实现）==========
    
    def encode(self, queries: List[str]) -> torch.Tensor:
        """
        编码query列表为embedding（可选）
        
        仅适用于基于相似度的路由模型（如DCRouterModel）
        
        Args:
            queries: query字符串列表
            
        Returns:
            shape: (batch_size, hidden_size)
            
        Raises:
            NotImplementedError: 如果模型不支持此方法
        """
        raise NotImplementedError("此模型不支持encode方法")
    
    def get_strategy_embeddings(self) -> Optional[torch.Tensor]:
        """
        获取策略embedding（可选）
        
        仅适用于基于相似度的路由模型（如DCRouterModel）
        
        Returns:
            shape: (num_strategies, hidden_size) 或 None
        """
        return None
    
    def compute_similarity(self, query_emb: torch.Tensor, strategy_embs: torch.Tensor) -> torch.Tensor:
        """
        计算query和策略embedding之间的相似度（可选）
        
        仅适用于基于相似度的路由模型（如DCRouterModel）
        
        Args:
            query_emb: query embedding, shape: (batch_size, hidden_size)
            strategy_embs: 策略embeddings, shape: (num_strategies, hidden_size)
            
        Returns:
            shape: (batch_size, num_strategies)
            
        Raises:
            NotImplementedError: 如果模型不支持此方法
        """
        raise NotImplementedError("此模型不支持相似度计算")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播（可选）
        
        注意：不同模型的forward签名可能不同
        - DCRouterModel: forward(input_ids, attention_mask) -> query_emb
        - FeatureFusedRouterModel: forward(input_ids, attention_mask, queries) -> logits
        
        Args:
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            **kwargs: 额外参数（如queries）
            
        Returns:
            根据模型类型不同，返回query embedding或logits
            
        Raises:
            NotImplementedError: 如果模型不支持此方法
        """
        raise NotImplementedError("此模型不支持forward方法")
