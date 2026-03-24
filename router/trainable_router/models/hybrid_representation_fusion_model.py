"""
混合表征融合路由器模型 - 使用 Cross Attention 融合内部表征和语义表征

架构：
1. 内部表征分支：预提取的 LLM 内部表征 (2048-d) → 投影层
2. 语义表征分支：MiniLM 编码器 (384-d, 可训练)
3. 融合层：Cross Attention (内部表征作为 Query, 语义表征作为 Key/Value)
4. 分类器：融合特征 → 策略预测
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import os
import json
import math

from ..base_model import BaseRouterModel
from ..factory import TrainableRouterFactory
from ..config import ModelConfig


class CrossAttentionFusion(nn.Module):
    """
    Cross Attention 融合模块
    
    使用内部表征作为 Query，语义表征作为 Key/Value
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,   # (B, L_q, D) - 内部表征
        key: torch.Tensor,     # (B, L_k, D) - 语义表征
        value: torch.Tensor,   # (B, L_v, D) - 语义表征
    ) -> torch.Tensor:
        """
        Cross Attention 前向传播
        
        Args:
            query: 内部表征 (B, L_q, D)
            key: 语义表征 (B, L_k, D)
            value: 语义表征 (B, L_v, D)
            
        Returns:
            融合后的特征 (B, L_q, D)
        """
        batch_size = query.size(0)
        
        # 投影
        Q = self.q_proj(query)  # (B, L_q, D)
        K = self.k_proj(key)    # (B, L_k, D)
        V = self.v_proj(value)  # (B, L_v, D)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_q, d)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_k, d)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_v, d)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L_q, L_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力
        attn_output = torch.matmul(attn_probs, V)  # (B, H, L_q, d)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        # 输出投影
        output = self.out_proj(attn_output)  # (B, L_q, D)
        
        return output


class BidirectionalCrossAttention(nn.Module):
    """
    双向 Cross Attention 融合模块
    
    同时计算两个方向的注意力：
    - 内部表征 → 语义表征
    - 语义表征 → 内部表征
    
    最后拼接或加权融合两个方向的输出
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_mode: str = "concat",  # "concat" or "add"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.fusion_mode = fusion_mode
        
        # 两个方向的 Cross Attention
        self.cross_attn_1 = CrossAttentionFusion(hidden_size, num_heads, dropout)  # internal -> semantic
        self.cross_attn_2 = CrossAttentionFusion(hidden_size, num_heads, dropout)  # semantic -> internal
        
        # 融合后的处理
        if fusion_mode == "concat":
            self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.output_proj = nn.Identity()
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        internal_rep: torch.Tensor,  # (B, L_i, D)
        semantic_rep: torch.Tensor,  # (B, L_s, D)
    ) -> torch.Tensor:
        """
        双向 Cross Attention
        
        Returns:
            融合后的特征 (B, max(L_i, L_s), D)
        """
        # 方向1: internal 作为 Query
        output_1 = self.cross_attn_1(query=internal_rep, key=semantic_rep, value=semantic_rep)
        
        # 方向2: semantic 作为 Query
        output_2 = self.cross_attn_2(query=semantic_rep, key=internal_rep, value=internal_rep)
        
        # 融合两个方向
        if self.fusion_mode == "concat":
            # 取第一个位置的输出进行拼接
            # 假设 L_i == L_s == 1
            fused = torch.cat([output_1, output_2], dim=-1)  # (B, L, 2*D)
            fused = self.output_proj(fused)  # (B, L, D)
        else:
            # 加权平均（需要相同长度）
            fused = (output_1 + output_2) / 2
        
        fused = self.layer_norm(fused)
        
        return fused


class HybridRepresentationFusionModel(BaseRouterModel):
    """
    混合表征融合路由器模型
    
    融合两种表征：
    1. LLM 内部表征 (frozen) - 深层语义信息
    2. MiniLM 语义表征 (trainable) - 通用语义信息
    
    融合方式：Cross Attention
    """
    
    def __init__(
        self, 
        config: ModelConfig,
        representation_dim: int = 2048,
        fusion_type: str = "cross_attn",  # "cross_attn" or "bidirectional_cross_attn"
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        freeze_internal_rep_proj: bool = False,
        freeze_backbone: bool = False,
        **kwargs
    ):
        """
        初始化
        
        Args:
            config: 模型配置
            representation_dim: 内部表征维度
            fusion_type: 融合方式 ("cross_attn" or "bidirectional_cross_attn")
            num_attention_heads: 注意力头数
            attention_dropout: 注意力 dropout
            freeze_internal_rep_proj: 是否冻结内部表征投影层
            freeze_backbone: 是否冻结 MiniLM backbone
            **kwargs: 额外参数
        """
        super().__init__(config)
        
        self.strategy_names = config.strategy_names
        self.num_strategies = config.num_strategies
        self.temperature = config.temperature
        
        # 从配置获取参数
        self.representation_dim = getattr(config, 'representation_dim', representation_dim)
        if hasattr(config, 'representation_dim') and config.representation_dim:
            self.representation_dim = config.representation_dim
        
        self.hidden_size = config.hidden_size
        self.fusion_type = fusion_type
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.freeze_internal_rep_proj = freeze_internal_rep_proj
        self.freeze_backbone = freeze_backbone
        
        # 初始化 MiniLM backbone
        self._init_backbone(config.backbone_name)
        
        # 构建网络
        self._build_network()
        
        # 冻结参数（如果需要）
        self._apply_freeze()
        
        # 初始化权重
        self._init_weights()
        
        print(f"HybridRepresentationFusionModel 初始化完成:")
        print(f"  - 内部表征维度: {self.representation_dim}")
        print(f"  - 语义表征维度: {self.semantic_dim}")
        print(f"  - 隐藏维度: {self.hidden_size}")
        print(f"  - 融合方式: {self.fusion_type}")
        print(f"  - 注意力头数: {self.num_attention_heads}")
        print(f"  - 策略数量: {self.num_strategies}")
        print(f"  - 冻结内部表征投影: {self.freeze_internal_rep_proj}")
        print(f"  - 冻结 Backbone: {self.freeze_backbone}")
    
    def _init_backbone(self, backbone_name: str):
        """
        初始化 MiniLM backbone
        
        Args:
            backbone_name: backbone 名称
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            self.backbone = AutoModel.from_pretrained(backbone_name)
            self.semantic_dim = self.backbone.config.hidden_size  # MiniLM: 384
            print(f'✓ 加载语义编码器: {backbone_name} (dim={self.semantic_dim})')
        except Exception as e:
            raise ImportError(f"无法加载模型 {backbone_name}: {e}")
    
    def _build_network(self):
        """构建网络结构"""
        # 内部表征投影层
        self.internal_proj = nn.Sequential(
            nn.Linear(self.representation_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 语义表征投影层
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.semantic_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Cross Attention 融合层
        if self.fusion_type == "cross_attn":
            self.fusion = nn.ModuleDict({
                'cross_attn': CrossAttentionFusion(
                    self.hidden_size,
                    self.num_attention_heads,
                    self.attention_dropout,
                ),
                'ffn': nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size * 4, self.hidden_size),
                ),
                'norm1': nn.LayerNorm(self.hidden_size),
                'norm2': nn.LayerNorm(self.hidden_size),
            })
        elif self.fusion_type == "bidirectional_cross_attn":
            self.fusion = BidirectionalCrossAttention(
                self.hidden_size,
                self.num_attention_heads,
                self.attention_dropout,
                fusion_mode="concat",
            )
        else:
            raise ValueError(f"不支持的融合方式: {self.fusion_type}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_strategies),
        )
    
    def _apply_freeze(self):
        """应用冻结策略"""
        # 冻结 backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("  ✓ Backbone 已冻结")
        
        # 冻结内部表征投影层
        if self.freeze_internal_rep_proj:
            for param in self.internal_proj.parameters():
                param.requires_grad = False
            print("  ✓ 内部表征投影层已冻结")
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.internal_proj, self.semantic_proj, self.classifier]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        编码 query 得到语义向量
        
        Args:
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            
        Returns:
            semantic embeddings, shape: (batch_size, semantic_dim)
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
            embeddings = last_hidden[:, 0, :]  # CLS token
        
        return embeddings
    
    def forward(
        self,
        representation: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            representation: 内部表征, shape: (batch_size, representation_dim)
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            
        Returns:
            logits, shape: (batch_size, num_strategies)
        """
        # 1. 投影内部表征
        internal_proj = self.internal_proj(representation)  # (B, D)
        internal_proj = internal_proj.unsqueeze(1)  # (B, 1, D) - 作为序列
        
        # 2. 编码语义表征
        semantic_emb = self.encode_query(input_ids, attention_mask)  # (B, semantic_dim)
        semantic_proj = self.semantic_proj(semantic_emb)  # (B, D)
        semantic_proj = semantic_proj.unsqueeze(1)  # (B, 1, D) - 作为序列
        
        # 3. Cross Attention 融合
        if self.fusion_type == "cross_attn":
            # 内部表征作为 Query，语义表征作为 Key/Value
            attn_output = self.fusion['cross_attn'](
                query=internal_proj,
                key=semantic_proj,
                value=semantic_proj,
            )  # (B, 1, D)
            
            # 残差连接 + LayerNorm
            fused = self.fusion['norm1'](internal_proj + attn_output)
            
            # FFN
            ffn_output = self.fusion['ffn'](fused)
            fused = self.fusion['norm2'](fused + ffn_output)  # (B, 1, D)
            
        elif self.fusion_type == "bidirectional_cross_attn":
            fused = self.fusion(internal_proj, semantic_proj)  # (B, 1, D)
        
        # 4. 分类
        fused = fused.squeeze(1)  # (B, D)
        logits = self.classifier(fused)  # (B, num_strategies)
        
        return logits
    
    # ========== 实现 BaseRouterModel 的方法 ==========
    
    def route(self, queries: List[str]) -> List[str]:
        """
        路由决策
        
        注意：此方法需要预提取的表征向量，不能仅凭 query 文本工作。
        请使用 route_with_representation() 方法。
        
        Args:
            queries: query 字符串列表
            
        Returns:
            策略名称列表
        """
        raise NotImplementedError(
            "HybridRepresentationFusionModel 需要预提取的表征向量。"
            "请使用 route_with_representation() 方法。"
        )
    
    def route_with_representation(
        self,
        representations: torch.Tensor,
        queries: List[str],
    ) -> List[str]:
        """
        使用表征向量和 query 文本进行路由决策
        
        Args:
            representations: 内部表征, shape: (batch_size, representation_dim)
            queries: query 字符串列表
            
        Returns:
            策略名称列表
        """
        self.eval()
        
        with torch.no_grad():
            representations = representations.to(self.device)
            
            # Tokenize
            inputs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 前向传播
            logits = self.forward(representations, input_ids, attention_mask)
            
            # 预测
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
                'model_type': 'hybrid_representation_fusion',
                'backbone_name': self.config.backbone_name,
                'representation_dim': self.representation_dim,
                'semantic_dim': self.semantic_dim,
                'hidden_size': self.hidden_size,
                'num_strategies': self.num_strategies,
                'fusion_type': self.fusion_type,
                'num_attention_heads': self.num_attention_heads,
                'attention_dropout': self.attention_dropout,
                'freeze_internal_rep_proj': self.freeze_internal_rep_proj,
                'freeze_backbone': self.freeze_backbone,
                'temperature': self.temperature,
            }
        }
        torch.save(model_state, os.path.join(path, 'model.pt'))
        
        # 保存配置
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'strategy_names': self.strategy_names,
                'backbone_name': self.config.backbone_name,
                'representation_dim': self.representation_dim,
                'semantic_dim': self.semantic_dim,
                'hidden_size': self.hidden_size,
                'num_strategies': self.num_strategies,
                'fusion_type': self.fusion_type,
                'num_attention_heads': self.num_attention_heads,
                'attention_dropout': self.attention_dropout,
                'freeze_internal_rep_proj': self.freeze_internal_rep_proj,
                'freeze_backbone': self.freeze_backbone,
                'temperature': self.temperature,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"模型已保存: {path}")
    
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
            if 'fusion_type' in config:
                self.fusion_type = config['fusion_type']
            if 'num_attention_heads' in config:
                self.num_attention_heads = config['num_attention_heads']
            if 'attention_dropout' in config:
                self.attention_dropout = config['attention_dropout']
        
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
        
        print(f"模型已加载: {path}")


# 注册到工厂
TrainableRouterFactory.register_model('hybrid_representation_fusion')(HybridRepresentationFusionModel)
TrainableRouterFactory.register_model('hybrid_rep_fusion')(HybridRepresentationFusionModel)
