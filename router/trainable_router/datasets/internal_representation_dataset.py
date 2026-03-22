"""
内部表征数据集 - 加载预提取的 LLM 内部表征

从 .npz 文件加载 LLM prefill 阶段的内部表征，用于训练路由器
"""

import json
import os
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from torch.utils.data import Dataset

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory


class InternalRepresentationDataset(BaseRouterDataset):
    """
    内部表征数据集
    
    从预提取的 .npz 文件加载 LLM 内部表征
    
    支持的表征类型：
    - shallow_mean: 第12层，masked mean pooling
    - shallow_last_token: 第12层，最后一个 token
    - middle_mean: 第24层，masked mean pooling
    - middle_last_token: 第24层，最后一个 token
    - deep_mean: 第36层，masked mean pooling
    - deep_last_token: 第36层，最后一个 token
    - next_token_logits: 最后位置的完整 logits
    
    数据格式：
    - metadata.json: 包含 question, optimal_strategy 等字段
    - shard_XXXX.npz: 包含表征向量，每个 shard 500 条
    """
    
    # 策略名称到索引的映射
    STRATEGY_TO_IDX = {
        'no_rag': 0,
        'naive_rag': 1,
        'graph_rag': 2,
    }
    
    def __init__(
        self, 
        config, 
        tokenizer=None,
        representation_type: str = 'deep_last_token',
        representation_dir: Optional[str] = None,
        **kwargs
    ):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器（此数据集不需要，保留接口兼容）
            representation_type: 表征类型，可选：
                - 'shallow_mean', 'shallow_last_token'
                - 'middle_mean', 'middle_last_token'
                - 'deep_mean', 'deep_last_token'
                - 'next_token_logits'
                - 'concat_deep': 拼接 deep_mean 和 deep_last_token
                - 'concat_all_mean': 拼接所有 mean pooling
                - 'concat_all_last': 拼接所有 last_token
            representation_dir: 表征目录路径（覆盖配置）
            **kwargs: 额外参数
        """
        super().__init__(config, tokenizer)
        
        self.representation_type = representation_type
        self.representation_dir = representation_dir
        
        # 获取策略名称列表
        self.strategy_names = config.strategy_names if hasattr(config, 'strategy_names') else ['no_rag', 'naive_rag']
        self.num_strategies = len(self.strategy_names)
        
        # 更新策略映射
        self.strategy_to_idx = {name: i for i, name in enumerate(self.strategy_names)}
        
        # 数据存储
        self.representations = []  # 表征向量列表
        self.labels = []           # 标签索引列表
        self.queries = []          # query 文本列表
        self.metadata = []         # 完整元数据
        
    def load_data(self, path: str):
        """
        加载数据
        
        Args:
            path: 数据目录路径（包含 metadata.json 和 shard_*.npz）
        """
        # 如果指定了 representation_dir，使用它
        data_dir = self.representation_dir or path
        
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        # 加载 metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"metadata.json 不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"加载 metadata: {len(self.metadata)} 条记录")
        
        # 加载所有 shard
        shard_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
        if not shard_files:
            raise ValueError(f"未找到 .npz 文件: {data_dir}")
        
        print(f"找到 {len(shard_files)} 个 shard 文件")
        
        # 加载表征向量
        all_representations = []
        for shard_file in shard_files:
            shard_path = os.path.join(data_dir, shard_file)
            shard_data = np.load(shard_path)
            
            # 根据表征类型选择向量
            if self.representation_type == 'concat_deep':
                # 拼接 deep_mean 和 deep_last_token
                rep = np.concatenate([
                    shard_data['deep_mean'],
                    shard_data['deep_last_token']
                ], axis=1)
            elif self.representation_type == 'concat_all_mean':
                # 拼接所有 mean pooling
                rep = np.concatenate([
                    shard_data['shallow_mean'],
                    shard_data['middle_mean'],
                    shard_data['deep_mean']
                ], axis=1)
            elif self.representation_type == 'concat_all_last':
                # 拼接所有 last_token
                rep = np.concatenate([
                    shard_data['shallow_last_token'],
                    shard_data['middle_last_token'],
                    shard_data['deep_last_token']
                ], axis=1)
            elif self.representation_type == 'concat_all':
                # 拼接所有表征（除了 logits）
                rep = np.concatenate([
                    shard_data['shallow_mean'],
                    shard_data['shallow_last_token'],
                    shard_data['middle_mean'],
                    shard_data['middle_last_token'],
                    shard_data['deep_mean'],
                    shard_data['deep_last_token']
                ], axis=1)
            else:
                # 单一表征类型
                if self.representation_type not in shard_data.files:
                    raise ValueError(
                        f"表征类型 '{self.representation_type}' 不存在。"
                        f"可用类型: {shard_data.files}"
                    )
                rep = shard_data[self.representation_type]
            
            all_representations.append(rep)
        
        # 合并所有 shard
        self.representations = np.concatenate(all_representations, axis=0)
        print(f"加载表征: shape={self.representations.shape}, dtype={self.representations.dtype}")
        
        # 提取标签和 query
        self.labels = []
        self.queries = []
        
        for i, item in enumerate(self.metadata):
            # 检查 optimal_strategy 是否存在
            if 'optimal_strategy' not in item:
                raise ValueError(
                    f"metadata[{i}] 缺少 'optimal_strategy' 字段: {item}"
                )
            
            strategy = item['optimal_strategy']
            if strategy not in self.strategy_to_idx:
                raise ValueError(
                    f"metadata[{i}] 包含未知策略 '{strategy}'，"
                    f"可用策略: {list(self.strategy_to_idx.keys())}"
                )
            
            # 检查 question 是否存在
            if 'question' not in item:
                raise ValueError(
                    f"metadata[{i}] 缺少 'question' 字段: {item}"
                )
            
            self.labels.append(self.strategy_to_idx[strategy])
            self.queries.append(item['question'])
        
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # 验证数据一致性
        if len(self.representations) != len(self.labels):
            raise ValueError(
                f"表征数量 ({len(self.representations)}) 与标签数量 ({len(self.labels)}) 不匹配"
            )
        
        print(f"数据集加载完成: {len(self)} 条样本")
        print(f"  - 表征维度: {self.representations.shape[1]}")
        print(f"  - 策略分布: {self._get_strategy_distribution()}")
    
    def _get_strategy_distribution(self) -> Dict[str, int]:
        """获取策略分布"""
        distribution = {}
        for i, name in enumerate(self.strategy_names):
            count = (self.labels == i).sum()
            distribution[name] = int(count)
        return distribution
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            字典，包含：
            - representation: 表征向量 (tensor)
            - label: 标签索引 (tensor)
            - query: query 文本 (str)
            - metadata: 完整元数据 (dict)
        """
        representation = torch.tensor(self.representations[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'representation': representation,
            'label': label,
            'query': self.queries[idx],
            'metadata': self.metadata[idx],
        }
    
    def get_representation_dim(self) -> int:
        """获取表征维度"""
        if len(self.representations) == 0:
            return 0
        return self.representations.shape[1]


# 注册到工厂
TrainableRouterFactory.register_dataset('internal_representation')(InternalRepresentationDataset)
TrainableRouterFactory.register_dataset('internal_rep')(InternalRepresentationDataset)
