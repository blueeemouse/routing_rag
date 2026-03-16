"""
融合模型软标签路由数据集

【重要】此类支持多分类任务，配合融合模型（如 GatedFusionModel）使用

与 BinarySoftLabelDataset 的区别：
================================================================================

| 特性             | BinarySoftLabelDataset      | FusionSoftLabelDataset         |
|------------------|----------------------------|--------------------------------|
| 分类类型         | 仅二分类                   | 支持多分类                     |
| 软标签格式       | 单值 (float)               | 向量 (List[float])             |
| 返回字段         | queries, soft_label        | input_ids, attention_mask, ... |
| 适用模型         | StatisticalRouterModel     | GatedFusionModel               |
| 损失函数         | BCEWithLogitsLoss          | CrossEntropyLoss               |

数据格式要求：
================================================================================

**二分类数据（兼容现有格式）**：
{
    "samples": [
        {
            "question": "...",
            "soft_label": 0.85,  # 单值，自动转换为向量 [0.15, 0.85]
            "label": "naive_rag",
            ...
        }
    ]
}

**多分类数据（推荐格式）**：
{
    "samples": [
        {
            "question": "...",
            "soft_label_vector": [0.1, 0.7, 0.2],  # 向量形式 [p_no_rag, p_naive_rag, p_graphrag]
            "label": "naive_rag",
            ...
        }
    ]
}

配合使用：
- FusionSoftLabelTrainer（使用 CrossEntropyLoss 支持软标签）
- GatedFusionModel（语义特征 + 统计特征融合）

================================================================================
"""

import os
import json
import math
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig
from ..data_utils import TrainingItem


class FusionSoftLabelDataset(BaseRouterDataset):
    """
    融合模型软标签路由数据集
    
    【特点】支持多分类，返回 tokenized 输入用于语义特征提取
    
    返回字段：
    - input_ids: token IDs (用于语义编码器)
    - attention_mask: 注意力掩码 (用于语义编码器)
    - queries: 原始问题文本 (用于统计特征提取)
    - soft_label: 软标签向量 (用于 CrossEntropyLoss)
    - label: 硬标签 (用于评估)
    """
    
    def __init__(
        self,
        config: TrainableRouterConfig,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
    ):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器 (用于语义特征提取)
            max_length: 最大序列长度
        """
        super().__init__(config, tokenizer)
        
        self.strategy_names = config.model.strategy_names
        self.num_strategies = config.model.num_strategies
        self.strategy_to_idx = {name: idx for idx, name in enumerate(self.strategy_names)}
        self.max_length = max_length
        self.raw_samples = []  # 存储原始数据
        
        # 如果没有提供 tokenizer，尝试从配置加载
        if self.tokenizer is None and hasattr(config.model, 'backbone_name'):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_name)
                print(f"FusionSoftLabelDataset 加载 tokenizer: {config.model.backbone_name}")
            except Exception as e:
                print(f"警告: 无法加载 tokenizer: {e}")
    
    def load_data(self, data_path: str):
        """
        加载软标签数据
        
        Args:
            data_path: 数据文件路径
        """
        if not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")
        
        print(f"加载融合软标签路由数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        print(f"找到 {len(samples)} 条样本")
        
        self.raw_samples = samples
        self.data = []
        
        # 统计信息
        no_soft_label_count = 0
        vector_format_count = 0
        single_value_count = 0
        
        for sample in samples:
            question = sample.get('question', '')
            if not question:
                continue
            
            # 构建 TrainingItem
            item = TrainingItem(
                question=question,
                strategy_scores={},
                cluster_id=sample.get('index', 0)
            )
            
            self.data.append(item)
            
            # 统计软标签格式
            if 'soft_label_vector' in sample:
                vector_format_count += 1
            elif 'soft_label' in sample:
                single_value_count += 1
            else:
                no_soft_label_count += 1
        
        print(f"成功加载 {len(self.data)} 条训练数据")
        print(f"  向量格式软标签: {vector_format_count}")
        print(f"  单值格式软标签: {single_value_count}")
        if no_soft_label_count > 0:
            print(f"  无软标签: {no_soft_label_count} (将从 em/f1 计算)")
        
        # 打印软标签分布统计
        self._print_soft_label_stats()
    
    def _print_soft_label_stats(self):
        """打印软标签分布统计"""
        if not self.raw_samples:
            return
        
        # 统计硬标签分布
        label_counts = {}
        for sample in self.raw_samples:
            label = sample.get('label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n硬标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({count/len(self.raw_samples)*100:.1f}%)")
        
        # 统计软标签分布（针对二分类）
        if self.num_strategies == 2:
            soft_labels = []
            for sample in self.raw_samples:
                sl = self._get_soft_label_vector(sample)
                if sl is not None:
                    # 取第二个值作为 naive_rag 的概率
                    soft_labels.append(sl[1] if len(sl) > 1 else sl[0])
            
            if soft_labels:
                near_0 = sum(1 for sl in soft_labels if sl < 0.3)
                near_05 = sum(1 for sl in soft_labels if 0.3 <= sl <= 0.7)
                near_1 = sum(1 for sl in soft_labels if sl > 0.7)
                
                print(f"\n软标签分布 (naive_rag 概率):")
                print(f"  < 0.3 (倾向 no_rag): {near_0} ({near_0/len(soft_labels)*100:.1f}%)")
                print(f"  0.3~0.7 (模糊): {near_05} ({near_05/len(soft_labels)*100:.1f}%)")
                print(f"  > 0.7 (倾向 naive_rag): {near_1} ({near_1/len(soft_labels)*100:.1f}%)")
    
    def _get_soft_label_vector(self, sample: Dict[str, Any]) -> Optional[List[float]]:
        """
        获取软标签向量
        
        支持两种格式：
        1. 向量格式: soft_label_vector = [p_0, p_1, ...]
        2. 单值格式: soft_label = p (二分类时转换为 [1-p, p])
        
        Args:
            sample: 原始数据样本
            
        Returns:
            软标签向量，或 None（需要计算）
        """
        # 优先使用向量格式
        if 'soft_label_vector' in sample:
            return sample['soft_label_vector']
        
        # 单值格式转换为向量
        if 'soft_label' in sample:
            p = sample['soft_label']
            if self.num_strategies == 2:
                # 二分类: [p_no_rag, p_naive_rag]
                return [1.0 - p, p]
            else:
                # 多分类: 不支持单值转向量
                print(f"警告: 多分类模式下不支持单值软标签，样本: {sample.get('question', '')[:50]}...")
                return None
        
        # 从 em/f1 计算软标签
        return self._compute_soft_label_from_metrics(sample)
    
    def _compute_soft_label_from_metrics(self, sample: Dict[str, Any]) -> List[float]:
        """
        从指标计算软标签向量
        
        使用 softmax over utility:
        utility_i = Q_i - cost_penalty_i
        soft_label = softmax(utilities)
        
        Args:
            sample: 原始数据样本
            
        Returns:
            软标签向量
        """
        utilities = []
        
        for strategy in self.strategy_names:
            em = sample.get(f'{strategy}_em', 0.0)
            f1 = sample.get(f'{strategy}_f1', 0.0)
            
            # Q = 0.8 * F1 + 0.2 * EM
            utility = 0.8 * f1 + 0.2 * em
            utilities.append(utility)
        
        # Softmax
        max_u = max(utilities)
        exp_u = [math.exp(u - max_u) for u in utilities]  # 数值稳定
        sum_exp = sum(exp_u)
        soft_label = [e / sum_exp for e in exp_u]
        
        return soft_label
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项
        
        Returns:
            {
                'input_ids': torch.Tensor,      # (max_length,)
                'attention_mask': torch.Tensor, # (max_length,)
                'queries': str,                 # 原始问题文本
                'soft_label': List[float],      # 软标签向量
                'label': int,                   # 硬标签索引
                'cluster_id': int,
            }
        """
        item = self.data[idx]
        raw_sample = self.raw_samples[idx] if idx < len(self.raw_samples) else {}
        
        # Tokenize
        input_ids = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                item.question,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 获取软标签向量
        soft_label = self._get_soft_label_vector(raw_sample)
        if soft_label is None:
            soft_label = [1.0 / self.num_strategies] * self.num_strategies  # 均匀分布作为 fallback
        
        # 获取硬标签（支持 'label' 或 'optimal_strategy' 字段）
        label_str = raw_sample.get('label') or raw_sample.get('optimal_strategy', '')
        label = self.strategy_to_idx.get(label_str, 0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'queries': item.question,
            'soft_label': soft_label,
            'label': label,
            'cluster_id': item.cluster_id,
        }
    
    def __len__(self) -> int:
        return len(self.data)


def fusion_soft_label_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate 函数，处理 soft_label 为向量格式
    
    Args:
        batch: 批次数据列表
        
    Returns:
        批次数据字典
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    queries = [item['queries'] for item in batch]
    soft_labels = torch.tensor([item['soft_label'] for item in batch], dtype=torch.float)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    cluster_ids = torch.tensor([item['cluster_id'] for item in batch], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'queries': queries,
        'soft_label': soft_labels,
        'label': labels,
        'cluster_id': cluster_ids,
    }


# 注册到工厂
TrainableRouterFactory.register_dataset('fusion_soft_label')(FusionSoftLabelDataset)
