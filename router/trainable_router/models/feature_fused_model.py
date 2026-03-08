"""
FeatureFusedRouterModel实现

基于语义特征+手工特征融合的路由模型
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import os
import json

from ..base_model import BaseRouterModel
from ..factory import TrainableRouterFactory
from ..config import ModelConfig
from ..feature_extraction import HandcraftedFeatureExtractor


class FeatureFusedRouterModel(BaseRouterModel):
    """
    特征融合路由模型
    
    流程：
    1. query → encoder → semantic_vec (768-d for BGE, 384-d for MiniLM)
    2. query → feature_extractor → handcrafted_vec (12-d)
    3. concat(semantic_vec, handcrafted_vec) → projection → classifier
    4. classifier → logits (num_strategies)
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
                - use_projection: 是否使用投影层（默认True）
        """
        super().__init__(config)
        
        self.strategy_names = config.strategy_names
        self.num_strategies = config.num_strategies
        self.temperature = config.temperature
        
        # 初始化backbone（语义编码器）
        self._init_backbone(config.backbone_name, **kwargs)
        
        # 手工特征配置
        self.use_spacy = kwargs.get('use_spacy', True)
        self.spacy_model = kwargs.get('spacy_model', 'en_core_web_sm')
        self.feature_normalize = kwargs.get('feature_normalize', True)
        
        # 初始化手工特征提取器
        self.feature_extractor = HandcraftedFeatureExtractor(
            use_spacy=self.use_spacy,
            spacy_model=self.spacy_model,
            normalize=self.feature_normalize
        )
        
        # 特征维度
        self.semantic_dim = self.hidden_size  # 768 (BGE) or 384 (MiniLM)
        self.handcrafted_dim = self.feature_extractor.get_feature_dimension()  # 63维
        self.fused_dim = self.semantic_dim + self.handcrafted_dim
        
        # 投影层配置
        self.use_projection = kwargs.get('use_projection', True)
        
        if self.use_projection:
            # 投影层：将融合特征降维到hidden_size
            self.projection = nn.Sequential(
                nn.Linear(self.fused_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            classifier_input_dim = self.hidden_size
        else:
            # 不使用投影层，直接用融合特征
            self.projection = None
            classifier_input_dim = self.fused_dim
        
        # 分类器：直接输出logits
        self.classifier = nn.Linear(classifier_input_dim, self.num_strategies)
        
        # 初始化权重
        self._init_weights()
        
        print(f"FeatureFusedRouterModel 初始化完成:")
        print(f"  - Semantic dim: {self.semantic_dim}")
        print(f"  - Handcrafted dim: {self.handcrafted_dim}")
        print(f"  - Fused dim: {self.fused_dim}")
        print(f"  - Use projection: {self.use_projection}")
        print(f"  - Classifier input dim: {classifier_input_dim}")
        print(f"  - Num strategies: {self.num_strategies}")
    
    def _init_backbone(self, backbone_name: str, **kwargs):
        """
        初始化backbone编码器
        
        Args:
            backbone_name: backbone名称
            **kwargs: 额外参数
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            self.backbone = AutoModel.from_pretrained(backbone_name)
            self.hidden_size = self.backbone.config.hidden_size
            print(f'✓ FeatureFusedRouterModel使用transformers加载: {backbone_name}')
        except Exception as e:
            raise ImportError(f"无法加载模型 {backbone_name}: {e}")
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化分类器
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # 初始化投影层（如果存在）
        if self.projection is not None:
            for module in self.projection:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        编码query得到语义向量
        
        Args:
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            
        Returns:
            semantic embeddings, shape: (batch_size, hidden_size)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        
        # Masked mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, L, 1)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / denom
        else:
            # Fallback to CLS token
            embeddings = last_hidden[:, 0, :]
        
        return embeddings
    
    def extract_handcrafted_features(self, queries: List[str]) -> torch.Tensor:
        """
        提取手工特征
        
        Args:
            queries: query字符串列表
            
        Returns:
            handcrafted features, shape: (batch_size, handcrafted_dim)
        """
        # 使用特征提取器批量提取
        features = self.feature_extractor.extract_batch(queries)
        
        # 移动到正确的设备
        return features.to(self.device)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        queries: List[str]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            queries: query字符串列表，用于提取手工特征
            
        Returns:
            logits, shape: (batch_size, num_strategies)
        """
        # 1. 提取语义特征
        semantic_features = self.encode_query(input_ids, attention_mask)  # (B, semantic_dim)
        
        # 2. 提取手工特征
        handcrafted_features = self.extract_handcrafted_features(queries)  # (B, handcrafted_dim)
        
        # 3. 拼接特征
        fused_features = torch.cat([semantic_features, handcrafted_features], dim=1)  # (B, fused_dim)
        
        # 4. 投影（可选）
        if self.projection is not None:
            projected_features = self.projection(fused_features)  # (B, hidden_size)
        else:
            projected_features = fused_features  # (B, fused_dim)
        
        # 5. 分类
        logits = self.classifier(projected_features)  # (B, num_strategies)
        
        return logits
    
    # ========== 实现BaseRouterModel的方法 ==========
    
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
            # 分词
            inputs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到设备
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 前向传播
            logits = self.forward(input_ids, attention_mask, queries)
            
            # 预测
            predicted_indices = logits.argmax(dim=-1)
            
            # 映射回策略名称
            routes = [self.strategy_names[idx.item()] for idx in predicted_indices]
        
        return routes
    
    def encode(self, queries: List[str]) -> torch.Tensor:
        """
        编码query列表为语义embedding（仅返回语义特征，不包含手工特征）
        
        Args:
            queries: query字符串列表
            
        Returns:
            semantic embeddings, shape: (batch_size, hidden_size)
        """
        self.eval()
        
        with torch.no_grad():
            # 分词
            inputs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 移动到设备
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 编码
            return self.encode_query(input_ids, attention_mask)
    
    def get_strategy_embeddings(self) -> Optional[torch.Tensor]:
        """
        获取策略embedding（FeatureFusedRouterModel不使用策略embedding）
        
        Returns:
            None（此模型不使用策略embedding机制）
        """
        return None
    
    def compute_similarity(self, query_emb: torch.Tensor, strategy_embs: torch.Tensor) -> torch.Tensor:
        """
        计算相似度（FeatureFusedRouterModel不使用相似度计算）
        
        Raises:
            NotImplementedError: 此模型使用分类器而非相似度计算
        """
        raise NotImplementedError(
            "FeatureFusedRouterModel does not use similarity-based routing. "
            "Use forward() method instead."
        )
    
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
                'model_type': 'feature_fused',
                'backbone_name': self.config.backbone_name,
                'hidden_size': self.hidden_size,
                'num_strategies': self.num_strategies,
                'use_spacy': self.use_spacy,
                'feature_normalize': self.feature_normalize,
                'use_projection': self.use_projection,
            }
        }
        torch.save(model_state, os.path.join(path, 'model.pt'))
        
        # 保存配置
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'strategy_names': self.strategy_names,
                'hidden_size': self.hidden_size,
                'num_strategies': self.num_strategies,
                'use_spacy': self.use_spacy,
                'feature_normalize': self.feature_normalize,
                'use_projection': self.use_projection,
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
        
        # 加载模型状态
        model_path = os.path.join(path, 'model.pt')
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=self.device, weights_only=False)

            # 兼容不同的保存格式
            if 'model_state_dict' in model_state:
                # 标准格式：trainer保存的checkpoint
                self.load_state_dict(model_state['model_state_dict'])
            elif 'state_dict' in model_state:
                # 旧格式：部分训练器使用的格式
                self.load_state_dict(model_state['state_dict'])
            else:
                # 直接保存的state_dict
                self.load_state_dict(model_state)


# 注册到工厂
TrainableRouterFactory.register_model('feature_fused')(FeatureFusedRouterModel)
