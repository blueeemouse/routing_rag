"""
软标签路由器数据集

支持软标签格式数据，兼容二分类和多分类场景。
可配合纯统计特征模型或融合模型使用。
"""

import os
import json
import math
from typing import Dict, Any, List, Optional, Union

import torch
from torch.utils.data import Dataset

from .base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig


class RouterSoftLabelDataset(BaseRouterDataset):
    """软标签路由器数据集
    
    支持的数据格式：
    1. 多分类软标签: soft_label_vector = [0.1, 0.7, 0.2]
    2. 二分类软标签: soft_label = 0.85 (自动转为 [0.15, 0.85])
    3. 无软标签: 从 em/f1 指标自动计算
    
    返回字段：
    - queries: str, 原始问题文本
    - soft_label: List[float], 软标签概率分布
    - label: int, 硬标签索引（从 soft_label argmax 推断）
    - cluster_id: int, 聚类标识
    - input_ids: torch.Tensor (optional), 语义特征
    - attention_mask: torch.Tensor (optional), 语义特征
    
    与 FusionSoftLabelDataset 的返回格式保持一致，便于后续替换。
    """
    
    def __init__(self, config: TrainableRouterConfig, split: str = "train"):
        """
        初始化
        
        Args:
            config: 配置对象
            split: 数据分割，"train"/"val"/"test"
        """
        super().__init__(config, split)
        
        # 统计信息
        self._print_soft_label_stats()
    
    def _process_samples(self) -> None:
        """处理样本（可选的额外处理）"""
        # 基类已加载数据到 self.data
        # 可以在这里添加数据验证或预处理
        pass
    
    def _print_soft_label_stats(self) -> None:
        """打印软标签统计信息"""
        if not self.data:
            return
        
        # 统计硬标签分布
        label_counts = {}
        for sample in self.data:
            label_str = sample.get('label') or sample.get('optimal_strategy', 'unknown')
            label_counts[label_str] = label_counts.get(label_str, 0) + 1
        
        print(f"\n硬标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({count/len(self.data)*100:.1f}%)")
        
        # 二分类时统计软标签分布
        if self.num_strategies == 2:
            soft_labels = []
            for sample in self.data:
                sl = self._get_soft_label_vector(sample)
                if sl and len(sl) > 1:
                    soft_labels.append(sl[1])  # naive_rag 概率
            
            if soft_labels:
                near_0 = sum(1 for sl in soft_labels if sl < 0.3)
                near_05 = sum(1 for sl in soft_labels if 0.3 <= sl <= 0.7)
                near_1 = sum(1 for sl in soft_labels if sl > 0.7)
                
                print(f"\n软标签分布 (naive_rag 概率):")
                print(f"  < 0.3 (倾向 no_rag): {near_0} ({near_0/len(soft_labels)*100:.1f}%)")
                print(f"  0.3~0.7 (模糊): {near_05} ({near_05/len(soft_labels)*100:.1f}%)")
                print(f"  > 0.7 (倾向 naive_rag): {near_1} ({near_1/len(soft_labels)*100:.1f}%)")
    
    def _get_soft_label_vector(self, sample: Dict[str, Any]) -> Optional[List[float]]:
        """获取软标签向量
        
        支持格式：
        1. soft_label_vector: [p_0, p_1, ...] （直接返回）
        2. soft_label: p （二分类时转为 [1-p, p]）
        3. 无软标签: 从 em/f1 计算
        
        Args:
            sample: 原始样本
            
        Returns:
            软标签概率分布，或 None（需要 fallback 计算）
        """
        # 优先使用向量格式
        if 'soft_label_vector' in sample:
            return sample['soft_label_vector']
        
        # 单值格式转为向量
        if 'soft_label' in sample:
            p = sample['soft_label']
            if self.num_strategies == 2:
                return [1.0 - p, p]
            else:
                # 多分类不支持单值转向量，使用 fallback
                print(f"警告: 多分类模式下不支持单值软标签")
                return None
        
        # 无软标签，需要从 em/f1 计算
        return None
    
    def _compute_soft_label_from_metrics(self, sample: Dict[str, Any]) -> List[float]:
        """从指标计算软标签向量
        
        使用 softmax over utility:
        utility_i = 0.8 * F1_i + 0.2 * EM_i
        soft_label = softmax(utilities)
        
        Args:
            sample: 原始样本
            
        Returns:
            软标签概率分布
        """
        utilities = []
        
        for strategy in self.strategy_names:
            em = sample.get(f'{strategy}_em', 0.0)
            f1 = sample.get(f'{strategy}_f1', 0.0)
            
            # Q = 0.8 * F1 + 0.2 * EM
            utility = 0.8 * f1 + 0.2 * em
            utilities.append(utility)
        
        # Softmax（数值稳定版本）
        max_u = max(utilities)
        exp_u = [math.exp(u - max_u) for u in utilities]
        sum_exp = sum(exp_u)
        
        if sum_exp == 0:
            # 均匀分布 fallback
            return [1.0 / self.num_strategies] * self.num_strategies
        
        return [e / sum_exp for e in exp_u]
    
    def get_soft_label(self, sample: Dict[str, Any]) -> List[float]:
        """获取软标签（公共接口）
        
        Args:
            sample: 原始样本
            
        Returns:
            软标签概率分布
        """
        soft_label = self._get_soft_label_vector(sample)
        if soft_label is None:
            soft_label = self._compute_soft_label_from_metrics(sample)
        return soft_label
    
    def get_label(self, sample: Dict[str, Any]) -> int:
        """获取硬标签索引（用于评估）
        
        优先从显式标签获取，否则从软标签 argmax 推断。
        
        Args:
            sample: 原始样本
            
        Returns:
            类别索引
        """
        # 优先使用显式硬标签
        label_str = sample.get('label') or sample.get('optimal_strategy')
        if label_str and label_str in self.strategy_to_idx:
            return self.strategy_to_idx[label_str]
        
        # 从软标签 argmax 推断
        soft_label = self.get_soft_label(sample)
        return soft_label.index(max(soft_label))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本
        
        返回格式与 FusionSoftLabelDataset 保持一致。
        
        Args:
            idx: 样本索引
            
        Returns:
            样本字典
        """
        sample = self.data[idx]
        question = sample.get('question', '')
        
        # 获取软标签和硬标签
        soft_label = self.get_soft_label(sample)
        label = self.get_label(sample)
        
        # 构建结果
        result = {
            'queries': question,
            'soft_label': soft_label,
            'label': label,
            'cluster_id': sample.get('index', idx)
        }
        
        # 语义特征（可选）
        if self.use_semantic and self.tokenizer:
            result.update(self._tokenize(question))
        
        return result


# 注册到工厂
TrainableRouterFactory.register_dataset('router_soft_label')(RouterSoftLabelDataset)
