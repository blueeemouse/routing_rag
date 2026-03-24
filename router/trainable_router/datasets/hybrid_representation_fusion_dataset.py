"""
混合表征融合数据集 - 同时加载内部表征和 Query 文本

数据源：
1. 内部表征：.npz 文件（预提取的 LLM 表征）
2. Query 文本 + 标签：JSON 文件

通过 question 字段匹配两种数据源
"""

import json
import os
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory


class HybridRepresentationFusionDataset(BaseRouterDataset):
    """
    混合表征融合数据集
    
    同时提供：
    - 内部表征向量（用于 Cross Attention 的 Query 分支）
    - Tokenized Query（用于语义编码器提取 Key/Value）
    - 标签（用于分类）
    
    数据匹配流程：
    1. 加载 .npz 文件获取内部表征
    2. 加载 JSON 文件获取 query 文本和标签
    3. 通过 question 字段匹配两者
    """
    
    def __init__(
        self,
        config,
        tokenizer: Optional[AutoTokenizer] = None,
        representation_type: str = 'deep_last_token',
        max_length: int = 512,
        **kwargs
    ):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器
            representation_type: 表征类型
            max_length: 最大序列长度
            **kwargs: 额外参数
        """
        super().__init__(config, tokenizer)
        
        self.representation_type = representation_type
        self.max_length = max_length
        
        # 获取策略名称列表
        self.strategy_names = config.strategy_names if hasattr(config, 'strategy_names') else ['no_rag', 'naive_rag']
        self.num_strategies = len(self.strategy_names)
        
        # 策略映射
        self.strategy_to_idx = {name: i for i, name in enumerate(self.strategy_names)}
        
        # 数据存储
        self.representations = []  # 内部表征向量
        self.queries = []          # query 文本
        self.labels = []           # 标签索引
        self.metadata = []         # 完整元数据
        
        # 数据路径（延迟加载）
        self.representation_dir = None
        self.labels_path = None
        
    def load_data(self, path: str, from_metadata: bool = False):
        """
        加载数据
        
        Args:
            path: 数据路径（表征目录路径）
            from_metadata: 是否强制从 metadata.json 加载标签（验证集用）
        """
        # path 参数优先级最高（用于验证集加载不同目录）
        self.representation_dir = path
        
        # 如果指定 from_metadata，禁用 labels_path
        if from_metadata:
            self.labels_path = None
        elif self.labels_path is None and hasattr(self.config, 'data'):
            # 否则从配置获取
            self.labels_path = getattr(self.config.data, 'labels_path', None)
        
        # 验证路径
        if not os.path.exists(self.representation_dir):
            raise ValueError(f"表征目录不存在: {self.representation_dir}")
        
        # 加载内部表征
        self._load_representations()
        
        # 加载标签和 query
        # 优先使用 labels_path（训练集），否则从 metadata 加载（验证集）
        if self.labels_path and os.path.exists(self.labels_path):
            self._load_labels_and_match()
        else:
            # 从表征目录的 metadata.json 加载
            self._load_from_metadata()
        
        print(f"数据集加载完成: {len(self)} 条样本")
        print(f"  - 表征维度: {self.representations.shape[1]}")
        print(f"  - 策略分布: {self._get_strategy_distribution()}")
    
    def _load_representations(self):
        """加载内部表征"""
        # 加载所有 shard
        shard_files = sorted([f for f in os.listdir(self.representation_dir) if f.endswith('.npz')])
        
        if not shard_files:
            raise ValueError(f"未找到 .npz 文件: {self.representation_dir}")
        
        print(f"找到 {len(shard_files)} 个表征 shard 文件")
        
        all_representations = []
        for shard_file in shard_files:
            shard_path = os.path.join(self.representation_dir, shard_file)
            shard_data = np.load(shard_path)
            
            # 根据表征类型选择向量
            if self.representation_type == 'concat_deep':
                rep = np.concatenate([
                    shard_data['deep_mean'],
                    shard_data['deep_last_token']
                ], axis=1)
            elif self.representation_type == 'concat_all_mean':
                rep = np.concatenate([
                    shard_data['shallow_mean'],
                    shard_data['middle_mean'],
                    shard_data['deep_mean']
                ], axis=1)
            elif self.representation_type == 'concat_all_last':
                rep = np.concatenate([
                    shard_data['shallow_last_token'],
                    shard_data['middle_last_token'],
                    shard_data['deep_last_token']
                ], axis=1)
            elif self.representation_type == 'concat_all':
                rep = np.concatenate([
                    shard_data['shallow_mean'],
                    shard_data['shallow_last_token'],
                    shard_data['middle_mean'],
                    shard_data['middle_last_token'],
                    shard_data['deep_mean'],
                    shard_data['deep_last_token']
                ], axis=1)
            else:
                if self.representation_type not in shard_data.files:
                    raise ValueError(
                        f"表征类型 '{self.representation_type}' 不存在。"
                        f"可用类型: {shard_data.files}"
                    )
                rep = shard_data[self.representation_type]
            
            all_representations.append(rep)
        
        self.representations = np.concatenate(all_representations, axis=0)
        print(f"加载表征: shape={self.representations.shape}, dtype={self.representations.dtype}")
    
    def _load_labels_and_match(self):
        """加载标签文件并与表征匹配"""
        print(f"加载标签文件: {self.labels_path}")
        
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        print(f"标签文件包含 {len(samples)} 条样本")
        
        # 构建 question -> 标签 的映射
        question_to_label = {}
        for sample in samples:
            question = sample.get('question', '').strip()
            if not question:
                continue
            question_to_label[question] = sample
        
        # 加载表征目录的 metadata.json 获取 question
        metadata_path = os.path.join(self.representation_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"metadata.json 不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            rep_metadata = json.load(f)
        
        print(f"表征 metadata 包含 {len(rep_metadata)} 条记录")
        
        # 匹配
        matched_count = 0
        self.queries = []
        self.labels = []
        self.metadata = []
        matched_indices = []
        
        for i, item in enumerate(rep_metadata):
            question = item.get('question', '').strip()
            if not question:
                continue
            
            if question in question_to_label:
                sample = question_to_label[question]
                strategy = sample.get('optimal_strategy', sample.get('label', ''))
                
                if strategy not in self.strategy_to_idx:
                    continue
                
                self.queries.append(question)
                self.labels.append(self.strategy_to_idx[strategy])
                self.metadata.append(sample)
                matched_indices.append(i)
                matched_count += 1
        
        # 过滤表征
        self.representations = self.representations[matched_indices]
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"匹配成功: {matched_count} 条样本")
    
    def _load_from_metadata(self):
        """从表征目录的 metadata.json 加载所有数据"""
        metadata_path = os.path.join(self.representation_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise ValueError(f"metadata.json 不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"从 metadata.json 加载 {len(self.metadata)} 条样本")
        
        self.queries = []
        self.labels = []
        valid_indices = []
        
        for i, item in enumerate(self.metadata):
            question = item.get('question', '')
            strategy = item.get('optimal_strategy', '')
            
            if not question or not strategy:
                continue
            
            if strategy not in self.strategy_to_idx:
                continue
            
            self.queries.append(question)
            self.labels.append(self.strategy_to_idx[strategy])
            valid_indices.append(i)
        
        # 过滤表征
        self.representations = self.representations[valid_indices]
        self.labels = np.array(self.labels, dtype=np.int64)
        self.metadata = [self.metadata[i] for i in valid_indices]
        
        print(f"有效样本: {len(self.queries)} 条")
    
    def _get_strategy_distribution(self) -> Dict[str, int]:
        """获取策略分布"""
        distribution = {}
        for i, name in enumerate(self.strategy_names):
            count = (self.labels == i).sum()
            distribution[name] = int(count)
        return distribution
    
    def get_representation_dim(self) -> int:
        """获取表征维度"""
        if len(self.representations) == 0:
            return 0
        return self.representations.shape[1]
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            字典，包含：
            - representation: 内部表征向量 (tensor)
            - input_ids: token IDs (tensor)
            - attention_mask: 注意力掩码 (tensor)
            - label: 标签索引 (tensor)
            - query: query 文本 (str)
        """
        representation = torch.tensor(self.representations[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        query = self.queries[idx]
        
        # Tokenize
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                query,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        return {
            'representation': representation,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'query': query,
        }


def hybrid_representation_fusion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate 函数
    
    Args:
        batch: 批次数据列表
        
    Returns:
        批次数据字典
    """
    representations = torch.stack([item['representation'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    queries = [item['query'] for item in batch]
    
    return {
        'representation': representations,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'queries': queries,
    }


# 注册到工厂
TrainableRouterFactory.register_dataset('hybrid_representation_fusion')(HybridRepresentationFusionDataset)
TrainableRouterFactory.register_dataset('hybrid_rep_fusion')(HybridRepresentationFusionDataset)
