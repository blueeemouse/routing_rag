"""
DecisionRouterModel实现

决策式路由器模型：
- 输入：query文本
- 输出：每个策略的性能预测(Q)和成本预测(cost)
- 决策：utility = Q - λ × cost，选择utility最高的策略

架构：
1. Backbone (BGE): query -> embedding
2. Q Head: embedding -> Q预测 (每个策略的预测值)
3. Cost Head: embedding -> cost预测 (每个策略的预测值)
4. Decision: utility = Q - λ × cost，argmax选择策略

训练目标：
- Q loss: MSE(Q_pred, Q_true)
- Cost loss: MSE(cost_pred, cost_true) (仅对需要检索的策略)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import os
import json
import numpy as np

from ..base_model import BaseRouterModel
from ..factory import TrainableRouterFactory
from ..config import ModelConfig


class DecisionRouterModel(BaseRouterModel):
    """
    决策式路由器模型
    
    核心思想：
    1. 预测每个策略的性能(Q)和成本(cost)
    2. 计算utility = Q - λ × cost
    3. 选择utility最高的策略
    
    支持两种模式：
    - 训练模式：输出Q_pred和cost_pred，用于计算回归损失
    - 推理模式：输出最终路由决策
    """
    
    def __init__(self, config: ModelConfig, **kwargs):
        """
        初始化
        
        Args:
            config: 模型配置
            **kwargs: 额外参数，支持：
                - lambda_cost: cost权重系数（默认1.0）
                - hidden_dropout_prob: dropout概率（默认0.1）
                - q_head_hidden_dim: Q head隐藏层维度（默认256）
                - cost_head_hidden_dim: Cost head隐藏层维度（默认128）
                - use_shared_hidden: 是否使用共享隐藏层（默认True）
                - shared_hidden_dim: 共享隐藏层维度（默认512）
        """
        super().__init__(config)
        
        self.strategy_names = config.strategy_names
        self.num_strategies = config.num_strategies
        self.temperature = config.temperature
        
        # 决策参数
        self.lambda_cost = kwargs.get('lambda_cost', 1.0)
        
        # 网络配置
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.q_head_hidden_dim = kwargs.get('q_head_hidden_dim', 256)
        self.cost_head_hidden_dim = kwargs.get('cost_head_hidden_dim', 128)
        self.use_shared_hidden = kwargs.get('use_shared_hidden', True)
        self.shared_hidden_dim = kwargs.get('shared_hidden_dim', 512)
        
        # 初始化backbone
        self._init_backbone(config.backbone_name, **kwargs)
        
        # 构建网络
        self._build_heads()
        
        # 初始化权重
        self._init_weights()
        
        # 记录哪些策略需要检索（有cost）
        # 默认：no_rag不需要检索，其他策略需要
        self.need_retrieval = {
            'no_rag': False,
            'naive_rag': True,
            'graph_rag': True,
        }
        
        print(f"DecisionRouterModel 初始化完成:")
        print(f"  - Backbone: {config.backbone_name}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Num strategies: {self.num_strategies}")
        print(f"  - Lambda cost: {self.lambda_cost}")
        print(f"  - Strategies: {self.strategy_names}")
        print(f"  - Use shared hidden: {self.use_shared_hidden}")
    
    def _init_backbone(self, backbone_name: str, **kwargs):
        """
        初始化backbone（使用transformers库）
        
        Args:
            backbone_name: backbone名称
            **kwargs: 额外参数
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            self.backbone = AutoModel.from_pretrained(backbone_name)
            self.hidden_size = self.backbone.config.hidden_size
            print(f'DecisionRouterModel 使用 transformers 加载: {backbone_name}')
        except Exception as e:
            raise ImportError(f"无法加载模型 {backbone_name}: {e}")
    
    def _build_heads(self):
        """构建预测头"""
        dropout = self.hidden_dropout_prob
        
        if self.use_shared_hidden:
            # 共享隐藏层
            self.shared_hidden = nn.Sequential(
                nn.Linear(self.hidden_size, self.shared_hidden_dim),
                nn.LayerNorm(self.shared_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            input_dim = self.shared_hidden_dim
        else:
            self.shared_hidden = None
            input_dim = self.hidden_size
        
        # Q预测头：预测每个策略的性能
        # 输出维度为 num_strategies
        self.q_head = nn.Sequential(
            nn.Linear(input_dim, self.q_head_hidden_dim),
            nn.LayerNorm(self.q_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.q_head_hidden_dim, self.q_head_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.q_head_hidden_dim // 2, self.num_strategies),
            nn.Sigmoid()  # Q值范围 [0, 1]
        )
        
        # Cost预测头：预测每个策略的归一化成本
        # 输出维度为 num_strategies
        self.cost_head = nn.Sequential(
            nn.Linear(input_dim, self.cost_head_hidden_dim),
            nn.LayerNorm(self.cost_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.cost_head_hidden_dim, self.cost_head_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.cost_head_hidden_dim // 2, self.num_strategies),
            nn.Sigmoid()  # cost值范围 [0, 1]
        )
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化共享层
        if self.shared_hidden is not None:
            for module in self.shared_hidden.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # 初始化Q头
        for module in self.q_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 初始化Cost头
        for module in self.cost_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, queries: List[str]) -> torch.Tensor:
        """
        编码query列表为embedding
        
        Args:
            queries: query字符串列表
            
        Returns:
            shape: (batch_size, hidden_size)
        """
        inputs = self.tokenize(queries)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        
        # Masked mean pooling
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / denom
        else:
            embeddings = last_hidden[:, 0, :]
        
        return embeddings
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """分词"""
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            tokenizer = self.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)
        
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            
        Returns:
            {
                'query_emb': torch.Tensor,    # (batch_size, hidden_size)
                'Q_pred': torch.Tensor,       # (batch_size, num_strategies)
                'cost_pred': torch.Tensor,    # (batch_size, num_strategies)
                'utility': torch.Tensor,      # (batch_size, num_strategies)
            }
        """
        # 编码query
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        query_emb = summed / denom  # (B, H)
        
        # 共享隐藏层
        if self.shared_hidden is not None:
            hidden = self.shared_hidden(query_emb)
        else:
            hidden = query_emb
        
        # 预测Q和cost
        Q_pred = self.q_head(hidden)  # (B, num_strategies)
        cost_pred = self.cost_head(hidden)  # (B, num_strategies)
        
        # 计算utility
        utility = Q_pred - self.lambda_cost * cost_pred
        
        return {
            'query_emb': query_emb,
            'Q_pred': Q_pred,
            'cost_pred': cost_pred,
            'utility': utility,
        }
    
    def forward_with_queries(self, queries: List[str]) -> Dict[str, torch.Tensor]:
        """
        从query文本直接预测
        
        Args:
            queries: query字符串列表
            
        Returns:
            预测结果字典
        """
        inputs = self.tokenize(queries)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.forward(inputs['input_ids'], inputs['attention_mask'])
    
    def route(self, queries: List[str]) -> List[str]:
        """
        路由决策：根据query选择最佳策略
        
        Args:
            queries: query字符串列表
            
        Returns:
            策略名称列表
        """
        self.eval()
        with torch.no_grad():
            results = self.forward_with_queries(queries)
            utility = results['utility']  # (B, num_strategies)
            
            # 选择utility最高的策略
            predicted_indices = utility.argmax(dim=-1)
            
            # 映射回策略名称
            routes = [self.strategy_names[idx.item()] for idx in predicted_indices]
        
        return routes
    
    def predict_with_details(self, queries: List[str]) -> Dict[str, Any]:
        """
        返回详细的预测结果
        
        Args:
            queries: query字符串列表
            
        Returns:
            {
                'routes': List[str],           # 最终路由决策
                'Q_pred': np.ndarray,          # Q预测值
                'cost_pred': np.ndarray,       # cost预测值
                'utility': np.ndarray,         # utility值
            }
        """
        self.eval()
        with torch.no_grad():
            results = self.forward_with_queries(queries)
            
            utility = results['utility']
            predicted_indices = utility.argmax(dim=-1)
            routes = [self.strategy_names[idx.item()] for idx in predicted_indices]
            
            return {
                'routes': routes,
                'Q_pred': results['Q_pred'].cpu().numpy(),
                'cost_pred': results['cost_pred'].cpu().numpy(),
                'utility': utility.cpu().numpy(),
            }
    
    def set_lambda_cost(self, lambda_cost: float):
        """
        设置cost权重系数
        
        Args:
            lambda_cost: 新的权重系数
        """
        self.lambda_cost = lambda_cost
    
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
                'model_type': 'decision_router',
                'hidden_size': self.hidden_size,
                'num_strategies': self.num_strategies,
                'lambda_cost': self.lambda_cost,
                'use_shared_hidden': self.use_shared_hidden,
                'shared_hidden_dim': self.shared_hidden_dim,
                'q_head_hidden_dim': self.q_head_hidden_dim,
                'cost_head_hidden_dim': self.cost_head_hidden_dim,
            }
        }
        torch.save(model_state, os.path.join(path, 'model.pt'))
        
        # 保存配置
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'strategy_names': self.strategy_names,
                'num_strategies': self.num_strategies,
                'hidden_size': self.hidden_size,
                'backbone_name': self.config.backbone_name,
                'lambda_cost': self.lambda_cost,
                'use_shared_hidden': self.use_shared_hidden,
                'shared_hidden_dim': self.shared_hidden_dim,
                'q_head_hidden_dim': self.q_head_hidden_dim,
                'cost_head_hidden_dim': self.cost_head_hidden_dim,
                'temperature': self.temperature,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"模型已保存到: {path}")
    
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
            
            if 'lambda_cost' in config:
                self.lambda_cost = config['lambda_cost']
            if 'use_shared_hidden' in config:
                self.use_shared_hidden = config['use_shared_hidden']
            if 'shared_hidden_dim' in config:
                self.shared_hidden_dim = config['shared_hidden_dim']
            if 'q_head_hidden_dim' in config:
                self.q_head_hidden_dim = config['q_head_hidden_dim']
            if 'cost_head_hidden_dim' in config:
                self.cost_head_hidden_dim = config['cost_head_hidden_dim']
        
        # 加载模型状态
        model_path = os.path.join(path, 'model.pt')
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in model_state:
                self.load_state_dict(model_state['model_state_dict'])
            elif 'state_dict' in model_state:
                self.load_state_dict(model_state['state_dict'])
            else:
                self.load_state_dict(model_state)
        
        print(f"模型已从: {path} 加载")


# 注册到工厂
TrainableRouterFactory.register_model('decision_router')(DecisionRouterModel)
