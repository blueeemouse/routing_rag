"""
支持样本权重的路由标签数据集

在RouterLabelDataset基础上，为每个样本添加权重
用于调整tie样本的训练权重
"""

import os
import json
from typing import Dict, Any, Set

from .router_label_dataset import RouterLabelDataset
from ..config import TrainableRouterConfig


class WeightedRouterLabelDataset(RouterLabelDataset):
    """
    支持样本权重的路由标签数据集
    
    在RouterLabelDataset基础上，为每个样本添加权重
    用于调整tie样本的训练权重
    
    示例:
        dataset = WeightedRouterLabelDataset(config, tie_weight=0.5)
        dataset.load_data('all_labels_with_tie_converted.json')
    """
    
    def __init__(
        self, 
        config: TrainableRouterConfig, 
        tokenizer=None,
        tie_weight: float = 0.5
    ):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器
            tie_weight: tie样本的权重（默认0.5）
        """
        super().__init__(config, tokenizer)
        self.tie_weight = tie_weight
        self.tie_sample_indices: Set[int] = set()  # 存储tie样本的索引
        self.raw_samples = []  # 存储原始样本数据（用于识别tie样本）
    
    def load_data(self, data_path: str):
        """
        加载数据，并识别tie样本
        
        Args:
            data_path: 数据文件路径
        """
        # 先保存原始样本数据
        if not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")
        
        print(f"加载加权路由标签数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.raw_samples = data.get('samples', [])
        
        # 调用父类的加载方法
        super().load_data(data_path)
        
        # 识别tie样本
        for idx, sample in enumerate(self.raw_samples):
            # 通过source字段判断是否是tie样本
            if sample.get('source') == 'tie_converted':
                self.tie_sample_indices.add(idx)
        
        # 打印权重信息
        num_tie = len(self.tie_sample_indices)
        num_normal = len(self.data) - num_tie
        print(f"\n样本权重分布:")
        print(f"  Tie样本: {num_tie} ({num_tie/len(self.data):.2%}) - 权重={self.tie_weight}")
        print(f"  正常样本: {num_normal} ({num_normal/len(self.data):.2%}) - 权重=1.0")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项，添加样本权重
        
        Args:
            idx: 索引
            
        Returns:
            数据项字典，包含sample_weight字段
        """
        # 调用父类的__getitem__获取基本数据
        result = super().__getitem__(idx)
        
        # 添加样本权重
        is_tie = idx in self.tie_sample_indices
        result['sample_weight'] = self.tie_weight if is_tie else 1.0
        
        return result
