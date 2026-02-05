"""
DCRouter模型实现

基于双对比学习的路由器模型
参考: LLMRouter/llmrouter/models/routerdc/dcmodel.py
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import os
import json

from ..base_model import BaseRouterModel
from ..factory import TrainableRouterFactory
from ..config import ModelConfig


class DCRouterModel(BaseRouterModel):
    """DCRouter模型
    包含一个query encoder以及候选策略的可学习的embedding，路由方式是用query经过编码的embedding与候选策略的embedding进行相似度计算，取得相似度最高的策略作为路由结果
    """
    
    def __init__(self, config: ModelConfig, **kwargs):
        """
        初始化
        
        Args:
            config: 模型配置
            **kwargs: 额外参数
        """
        super().__init__(config)
        
        self.strategy_names = config.strategy_names
        self.num_strategies = config.num_strategies
        self.similarity_function = config.similarity_function
        self.hidden_size = config.hidden_size
        self.temperature = config.temperature
        
        # 初始化backbone
        self._init_backbone(config.backbone_name, **kwargs)
        
        # 策略embedding (可学习的参数)
        self.strategy_embeddings = nn.Parameter(
            torch.randn(self.num_strategies, self.hidden_size)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_backbone(self, backbone_name: str, **kwargs):
        """
        初始化backbone encoder
        
        Args:
            backbone_name: backbone名称
            **kwargs: 额外参数
        """
        # 支持多种backbone
        if 'sentence-transformers' in backbone_name or 'all-MiniLM' in backbone_name:
            # 优先尝试使用 transformers 的 AutoModel + AutoTokenizer，以便 encoder 可训练
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
                self.backbone = AutoModel.from_pretrained(backbone_name)
                self.use_sentence_transformer = False
                print('DCRouterModel uses transformers AutoModel')     # 目前用的是这个
                if hasattr(self.backbone, 'config'):
                    self.hidden_size = self.backbone.config.hidden_size
            except Exception:
                # 回退到 sentence-transformers（兼容旧逻辑，但通常用于推理）
                try:
                    from sentence_transformers import SentenceTransformer
                    self.backbone = SentenceTransformer(backbone_name)
                    self.use_sentence_transformer = True
                    self.hidden_size = self.backbone.get_sentence_embedding_dimension()
                    print('use sentence-transformers')
                except ImportError:
                    raise ImportError("请安装 sentence-transformers 或 transformers: pip install sentence-transformers transformers")
        
        elif 'deberta' in backbone_name.lower() or 'mdeberta' in backbone_name.lower():
            try:
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(backbone_name)
                self.use_sentence_transformer = False
                print('use transformers AutoModel')
                # 获取隐藏层大小
                if hasattr(self.backbone, 'config'):
                    self.hidden_size = self.backbone.config.hidden_size
            except ImportError:
                raise ImportError("请安装transformers: pip install transformers")
        
        else:
            # 默认使用sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.backbone = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_sentence_transformer = True
                self.hidden_size = self.backbone.get_sentence_embedding_dimension()
                print('use sentence-transformers')
            except ImportError:
                raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
    
    def _init_weights(self):
        """初始化权重"""
        # 策略embedding使用较小的初始化
        nn.init.xavier_uniform_(self.strategy_embeddings)
    
    def encode(self, queries: List[str]) -> torch.Tensor:
        """
        编码query列表为embedding
        
        Args:
            queries: query字符串列表
            
        Returns:
            shape: (batch_size, hidden_size)
        """
        # 返回可导的 embeddings（允许梯度回传到 backbone）
        if self.use_sentence_transformer:
            # 兼容旧的 sentence-transformers（通常为回退情况）
            embeddings = self.backbone.encode(queries, convert_to_tensor=True)
            return embeddings.to(self.device)

        # 使用 transformers 的 AutoModel，并做 masked mean pooling
        inputs = self.tokenize(queries)
        # 将 inputs 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, L, 1)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / denom
        else:
            # fallback to CLS
            embeddings = last_hidden[:, 0, :]

        return embeddings
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """分词，优先使用已加载的 tokenizer"""
        from transformers import AutoTokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            tokenizer = self.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def get_strategy_embeddings(self) -> torch.Tensor:
        """
        获取策略embedding
        
        Returns:
            shape: (num_strategies, hidden_size)
        """
        return self.strategy_embeddings
    
    def compute_similarity(self, query_emb: torch.Tensor, strategy_embs: torch.Tensor) -> torch.Tensor:
        """
        计算query和策略embedding之间的相似度
        
        Args:
            query_emb: query embedding, shape: (batch_size, hidden_size)
            strategy_embs: 策略embeddings, shape: (num_strategies, hidden_size)
            
        Returns:
            shape: (batch_size, num_strategies)
        """
        if self.similarity_function == "cos":
            # Cosine similarity
            query_norm = torch.norm(query_emb, dim=1, keepdim=True)
            strategy_norm = torch.norm(strategy_embs, dim=1, keepdim=True)
            
            similarity = (query_emb @ strategy_embs.T) / (
                query_norm * strategy_norm.T + 1e-8
            )
        else:
            # Inner product
            similarity = query_emb @ strategy_embs.T
        
        # Apply temperature
        similarity = similarity / self.temperature
        
        return similarity
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播，完成对query的编码，返回query embedding
        
        Args:
            input_ids: token ids, shape: (batch_size, seq_len)
            attention_mask: 注意力掩码, shape: (batch_size, seq_len)
            
        Returns:
            query embeddings, shape: (batch_size, hidden_size)
        """
        if self.use_sentence_transformer:
            # SentenceTransformer需要字符串输入
            raise ValueError("SentenceTransformer backbone需要使用encode方法")

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # masked mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            query_emb = summed / denom
        else:
            query_emb = last_hidden[:, 0, :]

        return query_emb
    
    def forward_with_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回相似度分数
        
        Args:
            input_ids: token ids
            attention_mask: 注意力掩码
            
        Returns:
            包含query_emb和similarity的字典
        """
        query_emb = self.forward(input_ids, attention_mask)
        strategy_embs = self.get_strategy_embeddings()
        similarity = self.compute_similarity(query_emb, strategy_embs)
        
        return {
            'query_emb': query_emb,
            'similarity': similarity,
        }
    
    def compute_sample_llm_loss(
        self, 
        similarity: torch.Tensor, 
        index_true: torch.Tensor,
        top_k: int = 3,
        last_k: int = 3
    ) -> torch.Tensor:
        """
        计算Sample-LLM对比损失
        
        让query embedding更接近高分策略，更远离低分策略
        
        Args:
            similarity: 相似度矩阵, shape: (batch_size, num_strategies)
            index_true: 真实分数排序索引, shape: (batch_size, num_strategies)
            top_k: 前k个高分策略作为正样本
            last_k: 后k个低分策略作为负样本
            
        Returns:
            损失值
        """
        # 获取正样本索引
        top_index_true, _ = index_true.sort(dim=-1, descending=True)
        positive_indices = top_index_true[:, :top_k]  # (batch_size, top_k)
        
        # 获取负样本索引
        # 即使last_k=0，也要自动获取分数最低的策略作为负样本，防止塌缩
        if last_k > 0:
            last_index_true, _ = index_true.topk(k=last_k, largest=False)
            negative_indices = last_index_true  # (batch_size, last_k)
        else:
            # 自动获取最后1个（分数最低）作为负样本
            last_index_true, _ = index_true.topk(k=1, largest=False)
            negative_indices = last_index_true  # (batch_size, 1)
        
        # 计算损失
        loss = torch.tensor(0.0, device=self.device)
        sample_count = 0
        
        for i in range(similarity.size(0)):
            for pos_idx in positive_indices[i]:
                pos_score = similarity[i, pos_idx]
                
                if negative_indices is not None and negative_indices.size(1) > 0:
                    # 正样本对负样本的损失
                    for neg_idx in negative_indices[i]:
                        neg_score = similarity[i, neg_idx]
                        # log-sigmoid损失：鼓励pos_score > neg_score
                        loss += -torch.nn.functional.logsigmoid(pos_score - neg_score)
                        sample_count += 1
        
        if sample_count > 0:
            loss = loss / sample_count
        
        return loss
    
    def compute_sample_sample_loss(
        self, 
        query_embs: torch.Tensor, 
        cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Sample-Sample对比损失
        
        同一个cluster内的query应该有相似的表示
        
        Args:
            query_embs: query embeddings, shape: (batch_size, hidden_size)
            cluster_ids: cluster id, shape: (batch_size,)
            
        Returns:
            损失值
        """
        # Normalize embeddings
        query_embs = torch.nn.functional.normalize(query_embs, dim=1)

        cluster_ids = cluster_ids.to(query_embs.device)
        
        # 计算相似度矩阵
        sim_matrix = query_embs @ query_embs.T  # (batch_size, batch_size)
        
        # InfoNCE损失
        loss = torch.tensor(0.0, device=self.device)
        temp = 0.07
        
        for i in range(sim_matrix.size(0)):
            # 正样本：同一cluster的其他样本（不包括自己）
            positive_mask = (cluster_ids == cluster_ids[i]) & (torch.arange(sim_matrix.size(0), device=sim_matrix.device) != i)
            if not positive_mask.any():
                continue
            
            # 负样本：不同cluster的样本
            negative_mask = cluster_ids != cluster_ids[i]
            
            # 计算正样本的exp相似度
            pos_sim = torch.exp(sim_matrix[i][positive_mask] / temp)
            # 计算所有负样本的exp相似度之和
            neg_sim = torch.exp(sim_matrix[i][negative_mask] / temp).sum()
            
            # NCE损失
            loss += -torch.log(pos_sim / (pos_sim + neg_sim) + 1e-8).sum()
        
        return loss
    
    def compute_cluster_loss(
        self, 
        query_embs: torch.Tensor, 
        cluster_ids: torch.Tensor,
        num_clusters: int
    ) -> torch.Tensor:
        """
        计算Cluster对比损失
        
        Args:
            query_embs: query embeddings, shape: (batch_size, hidden_size)
            cluster_ids: cluster id, shape: (batch_size,)
            num_clusters: cluster总数
            
        Returns:
            损失值
        """
        # 计算每个cluster的中心
        cluster_centers = []
        for c in range(num_clusters):
            mask = cluster_ids == c
            if mask.any():
                center = query_embs[mask].mean(dim=0)
                cluster_centers.append(center)
        
        if len(cluster_centers) < 2:
            return torch.tensor(0.0, device=self.device)
        
        cluster_centers = torch.stack(cluster_centers)  # (num_used_clusters, hidden_size)
        cluster_centers = torch.nn.functional.normalize(cluster_centers, dim=1)
        
        # 计算cluster之间的相似度
        sim_matrix = cluster_centers @ cluster_centers.T
        
        # 对比损失：不同cluster应该更远
        temp = 0.07
        loss = torch.tensor(0.0, device=self.device)
        
        for i in range(len(cluster_centers)):
            # 正样本：自身
            pos_sim = torch.exp(sim_matrix[i, i] / temp)
            # 负样本：其他cluster
            neg_mask = torch.ones(len(cluster_centers), dtype=bool)
            neg_mask[i] = False
            neg_sim = torch.exp(sim_matrix[i][neg_mask] / temp).sum()
            
            loss += -torch.log(pos_sim / (pos_sim + neg_sim) + 1e-8)
        
        return loss
    
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
            # 编码query
            query_emb = self.encode(queries)
            
            # 获取策略embedding
            strategy_emb = self.get_strategy_embeddings()
            
            # 计算相似度
            similarity = self.compute_similarity(query_emb, strategy_emb)
            
            # 选择最高相似度的策略
            predicted_indices = similarity.argmax(dim=-1)
            
            # 映射回策略名称
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
                'model_type': 'dc',
                'similarity_function': self.similarity_function,
                'temperature': self.temperature,
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
            model_state = torch.load(model_path, map_location=self.device)
            self.load_state_dict(model_state['state_dict'])


# 注册到工厂（这里给同一个模型注册两个名字，倒也不怎么影响使用吧……可以认为兼容性比较好）
# 此时我们只要导入了这个类，就会完成注册
TrainableRouterFactory.register_model('dc')(DCRouterModel)
TrainableRouterFactory.register_model('dcrouter')(DCRouterModel)
