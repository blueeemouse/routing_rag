"""
StatisticalRouterModel实现

纯基于统计特征的路由器（参考ea_graphrag）
不依赖语义embedding，仅使用手工特征进行路由决策

核心思想：
- query → feature_extractor → handcrafted_features (85维)
- handcrafted_features → MLP Adapter → complexity_score (0-1)
- threshold → binary decision (no_rag vs naive_rag)
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
from ..feature_extraction import HandcraftedFeatureExtractor


class MLPMixer(nn.Module):
    """
    MLP Mixer模块（参考ea_graphrag的FeatureAttention）
    
    使用特征级注意力机制处理输入特征
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 46):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特征级注意力
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim)
        attn_weights = self.attention(x)  # (batch_size, input_dim)
        return x * attn_weights  # 注意力加权


class ResidualBlock(nn.Module):
    """
    残差块（参考ea_graphrag的Residual Blocks）
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        return self.relu(out + residual)


class StatisticalRouterModel(BaseRouterModel):
    """
    纯统计特征路由器
    
    架构（参考ea_graphrag）：
    1. 特征提取：query → HandcraftedFeatureExtractor → features (85维)
    2. 特征注意力：FeatureAttention
    3. MLP层：85 → 256 → 128 → 64
    4. 输出注意力：OutputAttention
    5. 分类器：64 → 1 (Sigmoid)
    
    路由决策：
    - complexity_score ≥ τ → naive_rag（复杂查询需要RAG）
    - complexity_score < τ → no_rag（简单查询不需要RAG）
    """
    
    def __init__(self, config: ModelConfig, **kwargs):
        """
        初始化
        
        Args:
            config: 模型配置
            **kwargs: 额外参数，支持：
                - use_spacy: 是否使用spaCy（默认True）
                - spacy_model: spaCy模型名称（默认'en_core_web_sm'）
                - feature_normalize: 是否归一化特征（默认True）
                - threshold: 路由阈值（默认0.5）
                - mlp_hidden_dims: MLP隐藏层维度列表（默认[256, 128, 64]）
                - dropout: dropout概率（默认0.1）
                - use_attention: 是否使用特征注意力（默认True）
                - use_residual: 是否使用残差连接（默认True）
                - label_smoothing: 标签平滑（默认0.1）
        """
        super().__init__(config)
        
        self.strategy_names = config.strategy_names  # ['no_rag', 'naive_rag']
        self.num_strategies = config.num_strategies  # 2
        self.temperature = config.temperature
        
        # 手工特征配置
        self.use_spacy = kwargs.get('use_spacy', True)
        self.spacy_model = kwargs.get('spacy_model', 'en_core_web_sm')
        self.feature_normalize = kwargs.get('feature_normalize', True)
        
        # 路由阈值
        self.threshold = kwargs.get('threshold', 0.5)
        
        # MLP配置
        self.mlp_hidden_dims = kwargs.get('mlp_hidden_dims', [256, 128, 64])
        self.dropout = kwargs.get('dropout', 0.1)
        self.use_attention = kwargs.get('use_attention', True)
        self.use_residual = kwargs.get('use_residual', True)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        
        # 初始化手工特征提取器
        self.feature_extractor = HandcraftedFeatureExtractor(
            use_spacy=self.use_spacy,
            spacy_model=self.spacy_model,
            normalize=self.feature_normalize
        )
        
        # 特征维度
        self.handcrafted_dim = self.feature_extractor.get_feature_dimension()  # 约85维
        input_dim = self.handcrafted_dim
        
        # ========== 网络架构 ==========
        
        # 特征级注意力
        if self.use_attention:
            self.feature_attention = MLPMixer(input_dim, hidden_dim=min(46, input_dim))
        else:
            self.feature_attention = None
        
        # MLP层（带残差连接）
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.mlp_hidden_dims:
            if self.use_residual and prev_dim == hidden_dim:
                layers.append(ResidualBlock(hidden_dim, self.dropout))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出层
        self.output = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self._init_weights()
        
        print(f"StatisticalRouterModel 初始化完成:")
        print(f"  - Feature dim: {self.handcrafted_dim}")
        print(f"  - MLP hidden dims: {self.mlp_hidden_dims}")
        print(f"  - Use attention: {self.use_attention}")
        print(f"  - Use residual: {self.use_residual}")
        print(f"  - Threshold: {self.threshold}")
        print(f"  - Strategies: {self.strategy_names}")
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化输出层
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        
        # 初始化注意力层
        if self.feature_attention is not None:
            for module in self.feature_attention.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # 初始化MLP层
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_features(self, queries: List[str]) -> torch.Tensor:
        """
        提取统计特征
        
        Args:
            queries: query字符串列表
            
        Returns:
            features, shape: (batch_size, handcrafted_dim)
        """
        # 使用特征提取器批量提取
        features = self.feature_extractor.extract_batch(queries)
        
        # 移动到正确的设备
        return features.to(self.device)
    
    def forward(self, queries: List[str]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            queries: query字符串列表
            
        Returns:
            logits, shape: (batch_size, 1) - 二分类logits
        """
        # 1. 提取统计特征
        features = self.extract_features(queries)  # (B, handcrafted_dim)
        
        # 2. 特征注意力
        if self.feature_attention is not None:
            features = self.feature_attention(features)
        
        # 3. MLP
        mlp_out = self.mlp(features)  # (B, last_hidden_dim)
        
        # 4. 输出层
        logits = self.output(mlp_out)  # (B, 1)
        
        return logits
    
    def get_probability(self, queries: List[str]) -> torch.Tensor:
        """
        获取属于naive_rag的概率
        
        Args:
            queries: query字符串列表
            
        Returns:
            probabilities, shape: (batch_size,)
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(queries)
            probs = torch.sigmoid(logits.squeeze(-1))
        
        return probs
    
    def get_complexity_score(self, queries: List[str]) -> np.ndarray:
        """
        获取复杂度分数（与ea_graphrag兼容）
        
        Args:
            queries: query字符串列表
            
        Returns:
            complexity scores, shape: (batch_size,)
        """
        probs = self.get_probability(queries)
        return probs.cpu().numpy()
    
    # ========== 实现BaseRouterModel的方法 ==========
    
    def route(self, queries: List[str]) -> List[str]:
        """
        路由决策：根据query选择最佳策略
        
        Args:
            queries: query字符串列表
            
        Returns:
            策略名称列表 ('no_rag' 或 'naive_rag')
        """
        self.eval()
        
        with torch.no_grad():
            # 获取复杂度分数
            probs = self.get_probability(queries)
            
            # 基于阈值决策
            # probability >= threshold → naive_rag (需要RAG)
            # probability < threshold → no_rag (不需要RAG)
            predicted_indices = (probs >= self.threshold).long()
            
            # 映射回策略名称
            routes = [self.strategy_names[idx.item()] for idx in predicted_indices]
        
        return routes
    
    def predict_with_scores(self, queries: List[str]) -> Dict[str, Any]:
        """
        返回详细的预测结果（包含分数）
        
        Args:
            queries: query字符串列表
            
        Returns:
            字典包含: routes, probabilities, complexity_scores
        """
        self.eval()
        
        with torch.no_grad():
            probs = self.get_probability(queries)
            
            predicted_indices = (probs >= self.threshold).long()
            routes = [self.strategy_names[idx.item()] for idx in predicted_indices]
            
            return {
                'routes': routes,
                'probabilities': probs.cpu().numpy(),
                'complexity_scores': probs.cpu().numpy(),  # 与ea_graphrag兼容
            }
    
    def set_threshold(self, threshold: float):
        """
        设置路由阈值
        
        Args:
            threshold: 新的阈值
        """
        self.threshold = threshold
    
    def optimize_threshold(
        self, 
        queries: List[str], 
        labels: List[int],
        metric: str = 'f1'
    ) -> float:
        """
        优化阈值
        
        Args:
            queries: query列表
            labels: 真实标签 (1=naive_rag, 0=no_rag)
            metric: 优化指标 ('f1', 'accuracy', 'balanced')
            
        Returns:
            最优阈值
        """
        self.eval()
        
        with torch.no_grad():
            probs = self.get_probability(queries).cpu().numpy()
            labels_np = np.array(labels)
            
            best_threshold = 0.5
            best_score = 0
            
            # 网格搜索
            for threshold in np.arange(0.1, 0.9, 0.05):
                preds = (probs >= threshold).astype(int)
                
                if metric == 'accuracy':
                    score = (preds == labels_np).mean()
                elif metric == 'f1':
                    tp = ((preds == 1) & (labels_np == 1)).sum()
                    fp = ((preds == 1) & (labels_np == 0)).sum()
                    fn = ((preds == 0) & (labels_np == 1)).sum()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    score = 2 * precision * recall / (precision + recall + 1e-8)
                elif metric == 'balanced':
                    # 平衡准确率
                    acc_0 = ((preds == 0) & (labels_np == 0)).sum() / max(1, (labels_np == 0).sum())
                    acc_1 = ((preds == 1) & (labels_np == 1)).sum() / max(1, (labels_np == 1).sum())
                    score = (acc_0 + acc_1) / 2
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            self.threshold = best_threshold
            return best_threshold
    
    # ========== 保存和加载 ==========
    
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
                'model_type': 'statistical',
                'num_strategies': self.num_strategies,
                'handcrafted_dim': self.handcrafted_dim,
                'use_spacy': self.use_spacy,
                'feature_normalize': self.feature_normalize,
                'threshold': self.threshold,
                'mlp_hidden_dims': self.mlp_hidden_dims,
                'dropout': self.dropout,
                'use_attention': self.use_attention,
                'use_residual': self.use_residual,
                'label_smoothing': self.label_smoothing,
            }
        }
        torch.save(model_state, os.path.join(path, 'model.pt'))
        
        # 保存配置（用于推理加载）
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'strategy_names': self.strategy_names,
                'num_strategies': self.num_strategies,
                'temperature': self.temperature,
                'use_spacy': self.use_spacy,
                'feature_normalize': self.feature_normalize,
                'threshold': self.threshold,
                'mlp_hidden_dims': self.mlp_hidden_dims,
                'dropout': self.dropout,
                'use_attention': self.use_attention,
                'use_residual': self.use_residual,
            }, f, indent=2, ensure_ascii=False)
        
        # 保存特征提取器的统计信息
        if self.feature_extractor.feature_stats is not None:
            stats_path = os.path.join(path, 'feature_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump({
                    k: (float(v[0]), float(v[1])) 
                    for k, v in self.feature_extractor.feature_stats.items()
                }, f, indent=2)
        
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
            
            if 'temperature' in config:
                self.temperature = config['temperature']
            if 'threshold' in config:
                self.threshold = config['threshold']
            if 'mlp_hidden_dims' in config:
                self.mlp_hidden_dims = config['mlp_hidden_dims']
            if 'dropout' in config:
                self.dropout = config['dropout']
            if 'use_attention' in config:
                self.use_attention = config['use_attention']
            if 'use_residual' in config:
                self.use_residual = config['use_residual']
            if 'use_spacy' in config:
                self.use_spacy = config['use_spacy']
            if 'feature_normalize' in config:
                self.feature_normalize = config['feature_normalize']
        
        # 加载特征统计信息
        stats_path = os.path.join(path, 'feature_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats_dict = json.load(f)
            self.feature_extractor.feature_stats = {
                k: tuple(v) for k, v in stats_dict.items()
            }
        
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
TrainableRouterFactory.register_model('statistical')(StatisticalRouterModel)
