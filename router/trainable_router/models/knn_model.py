"""
KNNRouter模型实现

基于K近邻的路由器模型
核心思想：将训练样本的embedding存储起来，推理时找到最近的k个邻居进行投票
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


class KNNRouterModel(BaseRouterModel):
    """KNNRouter模型
    
    基于K近邻的路由器，不需要梯度传播。
    训练阶段：存储所有训练样本的embedding和标签
    推理阶段：计算query的embedding，找最近的k个邻居投票决定路由
    
    Attributes:
        strategy_names: 策略名称列表
        num_strategies: 策略数量
        hidden_size: embedding维度
        k: KNN的k值
        distance_metric: 距离度量方式 ('cosine', 'euclidean')
        train_embeddings: 训练样本的embeddings (numpy array)
        train_labels: 训练样本的标签 (numpy array)
        is_fitted: 是否已经完成训练数据的embedding存储
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
        self.hidden_size = config.hidden_size
        self.temperature = config.temperature
        
        # KNN特有配置
        self.k = getattr(config, 'k', 5)  # 默认k=5
        self.distance_metric = getattr(config, 'distance_metric', 'cosine')  # 默认使用余弦距离
        self.weighted_voting = getattr(config, 'weighted_voting', True)  # 是否使用距离加权投票
        
        # 训练数据存储
        self.train_embeddings: Optional[np.ndarray] = None  # (num_samples, hidden_size)
        self.train_labels: Optional[np.ndarray] = None  # (num_samples,)
        self.train_queries: Optional[List[str]] = None  # 原始query文本，用于调试
        self.is_fitted = False
        
        # 初始化backbone (encoder)
        self._init_backbone(config.backbone_name, **kwargs)
    
    def _init_backbone(self, backbone_name: str, **kwargs):
        """
        初始化backbone encoder
        
        Args:
            backbone_name: backbone名称
            **kwargs: 额外参数
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            self.backbone = AutoModel.from_pretrained(backbone_name)
            # 更新hidden_size为backbone的实际维度
            self.hidden_size = self.backbone.config.hidden_size
            print(f'KNNRouterModel使用transformers加载: {backbone_name}')
            print(f'  Hidden size: {self.hidden_size}')
        except Exception as e:
            raise ImportError(f"无法加载模型 {backbone_name}: {e}")
    
    def encode(self, queries: List[str]) -> torch.Tensor:
        """
        编码query列表为embedding
        
        Args:
            queries: query字符串列表
            
        Returns:
            shape: (batch_size, hidden_size)
        """
        self.eval()
        with torch.no_grad():
            inputs = self.tokenize(queries)
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
                embeddings = last_hidden[:, 0, :]

        return embeddings
    
    def encode_batch(self, queries: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        批量编码query列表为embedding（用于训练阶段存储所有样本）
        
        Args:
            queries: query字符串列表
            batch_size: 批量大小
            show_progress: 是否显示进度条
            
        Returns:
            shape: (num_samples, hidden_size) numpy array
        """
        from tqdm import tqdm
        
        all_embeddings = []
        
        iterator = range(0, len(queries), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding queries", ascii=True)
        
        for i in iterator:
            batch_queries = queries[i:i + batch_size]
            batch_emb = self.encode(batch_queries)
            all_embeddings.append(batch_emb.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """分词"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def fit(self, queries: List[str], labels: List[int]) -> None:
        """
        拟合训练数据：存储所有样本的embedding和标签
        
        Args:
            queries: query字符串列表
            labels: 对应的标签列表
        """
        print(f"KNNRouter: 开始拟合 {len(queries)} 个训练样本...")
        
        # 编码所有训练样本
        self.train_embeddings = self.encode_batch(queries, show_progress=True)
        self.train_labels = np.array(labels, dtype=np.int64)
        self.train_queries = queries
        
        self.is_fitted = True
        
        # 打印统计信息
        unique_labels, counts = np.unique(self.train_labels, return_counts=True)
        print(f"KNNRouter: 拟合完成!")
        print(f"  总样本数: {len(queries)}")
        print(f"  Embedding维度: {self.hidden_size}")
        print(f"  标签分布:")
        for label, count in zip(unique_labels, counts):
            strategy_name = self.strategy_names[label] if label < len(self.strategy_names) else f"unknown_{label}"
            print(f"    {strategy_name}: {count} ({count/len(labels)*100:.1f}%)")
    
    def _compute_distances(self, query_emb: np.ndarray) -> np.ndarray:
        """
        计算query embedding与所有训练样本的距离
        
        Args:
            query_emb: 单个query的embedding, shape: (hidden_size,)
            
        Returns:
            shape: (num_train_samples,) 距离数组（值越小越相似）
        """
        if self.distance_metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            # 归一化
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            train_norms = self.train_embeddings / (np.linalg.norm(self.train_embeddings, axis=1, keepdims=True) + 1e-9)
            # 计算相似度
            similarities = np.dot(train_norms, query_norm)
            # 转换为距离
            distances = 1 - similarities
        elif self.distance_metric == 'euclidean':
            # 欧氏距离
            distances = np.linalg.norm(self.train_embeddings - query_emb, axis=1)
        else:
            raise ValueError(f"未知的距离度量方式: {self.distance_metric}")
        
        return distances
    
    def _predict_single(self, query_emb: np.ndarray) -> int:
        """
        预测单个query的标签
        
        Args:
            query_emb: 单个query的embedding, shape: (hidden_size,)
            
        Returns:
            预测的标签索引
        """
        # 计算距离
        distances = self._compute_distances(query_emb)
        
        # 找到最近的k个邻居
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.train_labels[k_nearest_indices]
        k_nearest_distances = distances[k_nearest_indices]
        
        # 投票
        if self.weighted_voting:
            # 距离加权投票：距离越近权重越大
            # 使用高斯核函数转换距离为权重
            weights = np.exp(-k_nearest_distances / self.temperature)
            
            # 计算每个类别的加权票数
            vote_counts = np.zeros(self.num_strategies)
            for label, weight in zip(k_nearest_labels, weights):
                vote_counts[label] += weight
            
            predicted_label = np.argmax(vote_counts)
        else:
            # 简单多数投票
            vote_counts = np.bincount(k_nearest_labels, minlength=self.num_strategies)
            predicted_label = np.argmax(vote_counts)
        
        return predicted_label
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播：编码query
        
        Args:
            input_ids: token ids
            attention_mask: 注意力掩码
            
        Returns:
            query embeddings
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            query_emb = summed / denom
        else:
            query_emb = last_hidden[:, 0, :]
        
        return query_emb
    
    def route(self, queries: List[str]) -> List[str]:
        """
        路由决策：根据query选择最佳策略
        
        Args:
            queries: query字符串列表
            
        Returns:
            策略名称列表
        """
        if not self.is_fitted:
            raise RuntimeError("KNNRouter尚未拟合训练数据，请先调用fit()方法或加载预训练模型")
        
        # 编码所有query
        query_embs = self.encode_batch(queries, show_progress=False)
        
        # 对每个query进行预测
        predictions = []
        for query_emb in query_embs:
            pred_label = self._predict_single(query_emb)
            predictions.append(pred_label)
        
        # 映射回策略名称
        routes = [self.strategy_names[idx] for idx in predictions]
        
        return routes
    
    def predict_proba(self, queries: List[str]) -> np.ndarray:
        """
        预测每个query属于各策略的概率分布
        
        Args:
            queries: query字符串列表
            
        Returns:
            shape: (num_queries, num_strategies) 概率分布
        """
        if not self.is_fitted:
            raise RuntimeError("KNNRouter尚未拟合训练数据，请先调用fit()方法或加载预训练模型")
        
        # 编码所有query
        query_embs = self.encode_batch(queries, show_progress=False)
        
        # 对每个query计算概率分布
        all_probas = []
        for query_emb in query_embs:
            distances = self._compute_distances(query_emb)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.train_labels[k_nearest_indices]
            k_nearest_distances = distances[k_nearest_indices]
            
            if self.weighted_voting:
                weights = np.exp(-k_nearest_distances / self.temperature)
                
                vote_counts = np.zeros(self.num_strategies)
                for label, weight in zip(k_nearest_labels, weights):
                    vote_counts[label] += weight
            else:
                vote_counts = np.bincount(k_nearest_labels, minlength=self.num_strategies)
            
            # 归一化为概率
            proba = vote_counts / (vote_counts.sum() + 1e-9)
            all_probas.append(proba)
        
        return np.array(all_probas)
    
    def get_knn_info(self, query: str) -> Dict[str, Any]:
        """
        获取单个query的KNN详细信息（用于调试和分析）
        
        Args:
            query: query字符串
            
        Returns:
            包含KNN信息的字典
        """
        if not self.is_fitted:
            raise RuntimeError("KNNRouter尚未拟合训练数据")
        
        query_emb = self.encode([query])[0].cpu().numpy()
        distances = self._compute_distances(query_emb)
        
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.train_labels[k_nearest_indices]
        k_nearest_distances = distances[k_nearest_indices]
        k_nearest_queries = [self.train_queries[i] for i in k_nearest_indices]
        
        return {
            'query': query,
            'k': self.k,
            'distance_metric': self.distance_metric,
            'neighbors': [
                {
                    'index': int(idx),
                    'query': q,
                    'label': int(label),
                    'strategy': self.strategy_names[label],
                    'distance': float(dist),
                }
                for idx, q, label, dist in zip(k_nearest_indices, k_nearest_queries, k_nearest_labels, k_nearest_distances)
            ],
            'predicted_strategy': self.strategy_names[self._predict_single(query_emb)],
        }
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存backbone
        backbone_path = os.path.join(path, 'backbone')
        self.backbone.save_pretrained(backbone_path)
        self.tokenizer.save_pretrained(backbone_path)
        
        # 保存训练数据（embedding和标签）
        if self.is_fitted:
            np.save(os.path.join(path, 'train_embeddings.npy'), self.train_embeddings)
            np.save(os.path.join(path, 'train_labels.npy'), self.train_labels)
            
            # 保存原始query（可选，用于调试）
            with open(os.path.join(path, 'train_queries.json'), 'w', encoding='utf-8') as f:
                json.dump(self.train_queries, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config_data = {
            'strategy_names': self.strategy_names,
            'num_strategies': self.num_strategies,
            'hidden_size': self.hidden_size,
            'k': self.k,
            'distance_metric': self.distance_metric,
            'weighted_voting': self.weighted_voting,
            'temperature': self.temperature,
            'is_fitted': self.is_fitted,
        }
        
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"KNNRouter模型已保存到: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        # 加载配置
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.strategy_names = config_data.get('strategy_names', self.strategy_names)
        self.num_strategies = config_data.get('num_strategies', len(self.strategy_names))
        self.k = config_data.get('k', self.k)
        self.distance_metric = config_data.get('distance_metric', self.distance_metric)
        self.weighted_voting = config_data.get('weighted_voting', self.weighted_voting)
        self.temperature = config_data.get('temperature', self.temperature)
        
        # 加载backbone
        backbone_path = os.path.join(path, 'backbone')
        if os.path.exists(backbone_path):
            from transformers import AutoModel, AutoTokenizer
            self.backbone = AutoModel.from_pretrained(backbone_path)
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_path)
            self.backbone.to(self.device)
        
        # 加载训练数据
        embeddings_path = os.path.join(path, 'train_embeddings.npy')
        labels_path = os.path.join(path, 'train_labels.npy')
        queries_path = os.path.join(path, 'train_queries.json')
        
        if os.path.exists(embeddings_path) and os.path.exists(labels_path):
            self.train_embeddings = np.load(embeddings_path)
            self.train_labels = np.load(labels_path)
            self.is_fitted = True
            
            if os.path.exists(queries_path):
                with open(queries_path, 'r', encoding='utf-8') as f:
                    self.train_queries = json.load(f)
            else:
                self.train_queries = None
            
            print(f"KNNRouter模型已加载: {len(self.train_labels)} 个训练样本")
        else:
            self.is_fitted = False
            print(f"KNNRouter模型已加载，但未找到训练数据")


# 注册到工厂
TrainableRouterFactory.register_model('knn')(KNNRouterModel)
