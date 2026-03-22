"""
内部表征路由器模型 - 基于 LLM 内部表征的分类器

使用预提取的 LLM 内部表征作为输入，通过 MLP 进行路由决策
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import os
import json

from ..base_model import BaseRouterModel
from ..factory import TrainableRouterFactory
from ..config import ModelConfig


class InternalRepresentationRouterModel(BaseRouterModel):
    """
    内部表征路由器模型
    
    架构：
    1. 输入：LLM 内部表征向量 (representation_dim)
    2. 投影层：representation_dim -> hidden_size
    3. 隐藏层：hidden_size -> hidden_size
    4. 分类器：hidden_size -> num_strategies
    
    支持的配置：
    - representation_dim: 输入表征维度（默认 2048）
    - hidden_size: 隐藏层维度
    - num_layers: 隐藏层数量
    - dropout: dropout 概率
    - use_layer_norm: 是否使用 LayerNorm
    """
    
    def __init__(
        self, 
        config: ModelConfig, 
        representation_dim: int = 2048,
        **kwargs
    ):
        """
        初始化
        
        Args:
            config: 模型配置
            representation_dim: 输入表征维度
            **kwargs: 额外参数
        """
        super().__init__(config)
        
        self.strategy_names = config.strategy_names
        self.num_strategies = config.num_strategies
        self.temperature = config.temperature
        
        # 从 kwargs 或 config 获取参数
        self.representation_dim = kwargs.get('representation_dim', representation_dim)
        if hasattr(config, 'representation_dim') and config.representation_dim:
            self.representation_dim = config.representation_dim
        
        self.hidden_size = config.hidden_size
        self.num_layers = kwargs.get('num_layers', 2)
        self.dropout = kwargs.get('dropout', 0.1)
        self.use_layer_norm = kwargs.get('use_layer_norm', True)
        
        # 构建网络
        self._build_network()
        
        # 初始化权重
        self._init_weights()
        
        print(f"InternalRepresentationRouterModel 初始化完成:")
        print(f"  - 输入维度: {self.representation_dim}")
        print(f"  - 隐藏维度: {self.hidden_size}")
        print(f"  - 隐藏层数: {self.num_layers}")
        print(f"  - 策略数量: {self.num_strategies}")
        print(f"  - Dropout: {self.dropout}")
    
    def _build_network(self):
        """构建网络结构"""
        layers = []
        
        # 输入投影层
        layers.append(nn.Linear(self.representation_dim, self.hidden_size))
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        # 隐藏层
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        # 分类器
        self.classifier = nn.Linear(self.hidden_size, self.num_strategies)
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            representation: 表征向量, shape: (batch_size, representation_dim)
            
        Returns:
            logits, shape: (batch_size, num_strategies)
        """
        # 编码
        hidden = self.encoder(representation)  # (B, hidden_size)
        
        # 分类
        logits = self.classifier(hidden)  # (B, num_strategies)
        
        return logits
    
    def predict_proba(self, representation: torch.Tensor) -> torch.Tensor:
        """
        预测概率分布
        
        Args:
            representation: 表征向量, shape: (batch_size, representation_dim)
            
        Returns:
            概率分布, shape: (batch_size, num_strategies)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(representation)
            probs = torch.softmax(logits / self.temperature, dim=-1)
        return probs
    
    # ========== 实现 BaseRouterModel 的方法 ==========
    
    def route(self, queries: List[str]) -> List[str]:
        """
        路由决策
        
        注意：此方法需要预先提取的表征向量，不能仅凭 query 文本工作。
        请使用 route_with_representation() 方法。
        
        Args:
            queries: query 字符串列表
            
        Returns:
            策略名称列表
        """
        raise NotImplementedError(
            "InternalRepresentationRouterModel 需要预提取的表征向量。"
            "请使用 route_with_representation() 方法。"
        )
    
    def route_with_representation(self, representations: torch.Tensor) -> List[str]:
        """
        使用表征向量进行路由决策
        
        Args:
            representations: 表征向量, shape: (batch_size, representation_dim)
            
        Returns:
            策略名称列表
        """
        self.eval()
        
        with torch.no_grad():
            representations = representations.to(self.device)
            logits = self.forward(representations)
            predicted_indices = logits.argmax(dim=-1)
            
            routes = [self.strategy_names[idx.item()] for idx in predicted_indices]
        
        return routes
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存模型状态
        model_state = {
            'state_dict': self.state_dict(),
            'strategy_names': self.strategy_names,
            'config': {
                'model_type': 'internal_representation',
                'representation_dim': self.representation_dim,
                'hidden_size': self.hidden_size,
                'num_strategies': self.num_strategies,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'use_layer_norm': self.use_layer_norm,
                'temperature': self.temperature,
            }
        }
        torch.save(model_state, os.path.join(path, 'model.pt'))
        
        # 保存配置（用于推理加载）
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'strategy_names': self.strategy_names,
                'representation_dim': self.representation_dim,
                'hidden_size': self.hidden_size,
                'num_strategies': self.num_strategies,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'use_layer_norm': self.use_layer_norm,
                'temperature': self.temperature,
            }, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        # 加载配置
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.strategy_names = config.get('strategy_names', self.strategy_names)
            self.num_strategies = config.get('num_strategies', len(self.strategy_names))
            if 'representation_dim' in config:
                self.representation_dim = config['representation_dim']
            if 'hidden_size' in config:
                self.hidden_size = config['hidden_size']
            if 'num_layers' in config:
                self.num_layers = config['num_layers']
            if 'dropout' in config:
                self.dropout = config['dropout']
            if 'use_layer_norm' in config:
                self.use_layer_norm = config['use_layer_norm']
            if 'temperature' in config:
                self.temperature = config['temperature']
        
        # 加载模型状态
        model_path = os.path.join(path, 'model.pt')
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 兼容不同的保存格式
            if 'model_state_dict' in model_state:
                self.load_state_dict(model_state['model_state_dict'])
            elif 'state_dict' in model_state:
                self.load_state_dict(model_state['state_dict'])
            else:
                self.load_state_dict(model_state)


# 注册到工厂
TrainableRouterFactory.register_model('internal_representation')(InternalRepresentationRouterModel)
TrainableRouterFactory.register_model('internal_rep')(InternalRepresentationRouterModel)
TrainableRouterFactory.register_model('representation')(InternalRepresentationRouterModel)
