"""
DCRouter推理类

用于推理阶段的路由器实现
"""

import os
import json
from typing import List, Dict, Any, Optional
import torch
import numpy as np

# 使用绝对导入（因为 train_router.py 已经将项目根目录添加到 sys.path）
from interfaces.router_interface import RouterInterface
from ..models.dc_model import DCRouterModel
from ..config import TrainableRouterConfig


class DCRouter(RouterInterface):
    """DCRouter推理类"""
    
    def __init__(
        self, 
        model_path: str,
        config: Optional[TrainableRouterConfig] = None,
        **kwargs
    ):
        """
        初始化
        
        Args:
            model_path: 训练好的模型路径
            config: 训练配置（可选）
            **kwargs: 额外参数
        """
        self.model_path = model_path
        self.config = config
        
        # 加载模型配置
        self._load_config()
        
        # 初始化模型
        self._init_model(**kwargs)
        
        # 加载权重
        self._load_weights()
    
    def _load_config(self):
        """加载配置"""
        if self.config is not None:
            return
        
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            from ..config import ModelConfig
            self.model_config = ModelConfig(
                backbone_name=raw_config.get('backbone_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                hidden_size=raw_config.get('hidden_size', 384),
                strategy_names=raw_config.get('strategy_names', ['no_rag', 'naive_rag', 'graph_rag']),
                num_strategies=raw_config.get('num_strategies', 3),
                similarity_function=raw_config.get('similarity_function', 'cos'),
                temperature=raw_config.get('temperature', 1.0),
                device='auto',
            )
        else:
            from ..config import ModelConfig
            self.model_config = ModelConfig(
                strategy_names=['no_rag', 'naive_rag', 'graph_rag'],
                num_strategies=3,
            )
    
    def _init_model(self, **kwargs):
        """初始化模型"""
        self.model = DCRouterModel(self.model_config, **kwargs)
        self.model.eval()
    
    def _load_weights(self):
        """加载权重"""
        model_file = os.path.join(self.model_path, 'model.pt')
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            # 兼容不同的保存格式
            if 'model_state_dict' in checkpoint:
                # 标准格式：trainer保存的checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # 旧格式：部分训练器使用的格式
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接保存的state_dict
                self.model.load_state_dict(checkpoint)
            
            print(f"已加载模型权重: {self.model_path}")
    
    def route(self, sub_query: str) -> str:
        """
        路由决策：根据query选择最佳策略
        
        Args:
            sub_query: 子查询字符串
            
        Returns:
            策略名称: 'no_rag' | 'naive_rag' | 'graph_rag'
        """
        routes = self.route_batch([sub_query])
        return routes[0]
    
    def route_batch(self, queries: List[str]) -> List[str]:
        """
        批量路由决策
        
        Args:
            queries: query字符串列表
            
        Returns:
            策略名称列表
        """
        self.model.eval()
        
        with torch.no_grad():
            # 编码query
            query_emb = self.model.encode(queries)
            
            # 获取策略embedding
            strategy_emb = self.model.get_strategy_embeddings()
            
            # 计算相似度
            similarity = self.model.compute_similarity(query_emb, strategy_emb)
            
            # 选择最高相似度的策略
            predicted_indices = similarity.argmax(dim=-1)
            
            # 映射回策略名称
            routes = [self.model.strategy_names[idx.item()] for idx in predicted_indices]
        
        return routes
    
    def route_with_scores(self, queries: List[str]) -> Dict[str, Any]:
        """
        路由决策并返回相似度分数
        
        Args:
            queries: query字符串列表
            
        Returns:
            包含路由结果和分数的字典
        """
        self.model.eval()
        
        with torch.no_grad():
            # 编码query
            query_emb = self.model.encode(queries)
            
            # 获取策略embedding
            strategy_emb = self.model.get_strategy_embeddings()
            
            # 计算相似度
            similarity = self.model.compute_similarity(query_emb, strategy_emb)
            similarity = torch.nn.functional.softmax(similarity, dim=-1)
            
            # 预测
            predicted_indices = similarity.argmax(dim=-1)
            probabilities = similarity.cpu().tolist()
            
            # 构建结果
            results = {
                'queries': queries,
                'predictions': [],
                'probabilities': probabilities,
            }
            
            for i, query in enumerate(queries):
                probs = {self.model.strategy_names[j]: probabilities[i][j] for j in range(len(self.model.strategy_names))}
                results['predictions'].append({
                    'query': query,
                    'route': self.model.strategy_names[predicted_indices[i].item()],
                    'probabilities': probs,
                })
        
        return results
    
    def get_strategy_embeddings(self) -> np.ndarray:
        """获取策略embedding"""
        with torch.no_grad():
            emb = self.model.get_strategy_embeddings().cpu().numpy()
        return emb
    
    def save(self, path: str):
        """保存模型"""
        self.model.save(path)
    
    def to(self, device: str):
        """移动模型到设备"""
        self.model.to(device)
    
    @classmethod
    def from_config(cls, config: TrainableRouterConfig) -> 'DCRouter':
        """
        从配置创建路由器
        
        Args:
            config: 配置
            
        Returns:
            DCRouter实例
        """
        model_path = config.save_model_path if config.save_model_path else config.output_dir
        return cls(model_path, config)
    
    @property
    def strategy_names(self) -> List[str]:
        """获取策略名称列表"""
        return self.model.strategy_names if hasattr(self.model, 'strategy_names') else []
    
    @property
    def device(self) -> torch.device:
        """获取设备"""
        return next(self.model.parameters()).device
