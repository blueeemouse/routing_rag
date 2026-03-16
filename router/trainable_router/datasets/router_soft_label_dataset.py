"""
软标签路由器数据集

支持软标签格式数据，兼容二分类和多分类场景。
可配合纯统计特征模型或融合模型使用。
"""

from typing import Dict, Any, List

import torch

from .base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig


class RouterSoftLabelDataset(BaseRouterDataset):
    """软标签路由器数据集
    
    支持的数据格式：
    1. 多分类软标签: soft_label_vector = [0.1, 0.7, 0.2]
    2. 二分类软标签: soft_label = 0.85 (自动转为 [0.15, 0.85])
    
    必需字段：
    - question: 问题文本
    - optimal_strategy: 最优策略名称（用于硬标签）
    - soft_label 或 soft_label_vector: 软标签（二选一）
    
    返回字段：
    - queries: str, 原始问题文本
    - soft_label: List[float], 软标签概率分布
    - label: int, 硬标签索引
    - cluster_id: int, 聚类标识
    - input_ids: torch.Tensor (optional), 语义特征
    - attention_mask: torch.Tensor (optional), 语义特征
    """
    
    def __init__(self, config: TrainableRouterConfig, split: str = "train"):
        """
        初始化
        
        Args:
            config: 配置对象
            split: 数据分割，"train"/"val"/"test"
        """
        super().__init__(config, split)
    
    def _get_soft_label(self, sample: Dict[str, Any]) -> List[float]:
        """获取软标签向量
        
        支持格式：
        1. soft_label_vector: [p_0, p_1, ...] （多分类，直接返回）
        2. soft_label: p （二分类时转为 [1-p, p]）
        
        Args:
            sample: 原始样本
            
        Returns:
            软标签概率分布
            
        Raises:
            KeyError: 如果找不到软标签字段
            ValueError: 如果格式不匹配
        """
        # 优先使用向量格式（多分类）
        if 'soft_label_vector' in sample:
            return sample['soft_label_vector']
        
        # 单值格式（二分类）
        if 'soft_label' in sample:
            p = sample['soft_label']
            if self.num_strategies == 2:
                return [1.0 - p, p]
            else:
                raise ValueError(
                    f"多分类模式下不支持单值软标签 (soft_label={p})，"
                    f"请使用 soft_label_vector 格式"
                )
        
        # 没有软标签字段，报错
        raise KeyError(
            f"样本缺少软标签字段 (soft_label 或 soft_label_vector)，"
            f"sample keys: {list(sample.keys())}"
        )
    
    def get_label(self, sample: Dict[str, Any]) -> int:
        """获取硬标签索引
        
        只认 optimal_strategy 字段，没有则报错。
        
        Args:
            sample: 原始样本
            
        Returns:
            类别索引
            
        Raises:
            KeyError: 如果没有 optimal_strategy 字段
            ValueError: 如果策略名不在配置中
        """
        label_str = sample['optimal_strategy']  # 没有就 KeyError
        
        if label_str not in self.strategy_to_idx:
            raise ValueError(
                f"未知的策略名称: {label_str}，"
                f"配置中的策略: {self.strategy_names}"
            )
        
        return self.strategy_to_idx[label_str]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本字典
            
        Raises:
            KeyError: 如果缺少必需字段
        """
        sample = self.data[idx]
        question = sample['question']  # 没有就 KeyError
        
        # 获取软标签和硬标签
        soft_label = self._get_soft_label(sample)
        label = self.get_label(sample)
        
        # 构建结果
        result = {
            'queries': question,
            'soft_label': soft_label,
            'label': label,
            'cluster_id': sample.get('cluster_id', 0)  # 默认0表示无聚类信息
        }
        
        # 语义特征（可选）
        if self.use_semantic and self.tokenizer:
            result.update(self._tokenize(question))
        
        return result


# 注册到工厂
TrainableRouterFactory.register_dataset('router_soft_label')(RouterSoftLabelDataset)
