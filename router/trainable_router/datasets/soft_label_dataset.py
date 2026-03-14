"""
软标签路由数据集

支持软标签训练，返回 soft_label 而非硬标签
用于处理 tie 样本和模糊边界情况
"""

import os
import json
from typing import Dict, Any, List

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig
from ..data_utils import TrainingItem


class SoftLabelRouterDataset(BaseRouterDataset):
    """
    软标签路由数据集
    
    与 RouterLabelDataset 的区别:
    - 返回 soft_label (0~1 的连续值) 而非硬标签
    - soft_label 含义: 接近 0 = no_rag 好, 接近 1 = naive_rag 好
    
    数据格式要求:
    {
        "samples": [
            {
                "question": "...",
                "soft_label": 0.85,  # 0~1 的软标签
                "utility_gap": 0.2,  # 可选, ΔU 值
                "no_rag_em": 0.0,    # 可选
                "no_rag_f1": 0.0,    # 可选
                "naive_rag_em": 1.0, # 可选
                "naive_rag_f1": 1.0  # 可选
            }
        ]
    }
    """
    
    def __init__(
        self,
        config: TrainableRouterConfig,
        tokenizer=None,
        soft_label_threshold: float = 0.5,  # 软标签阈值，用于评估时的预测
    ):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器
            soft_label_threshold: 预测阈值，soft_label > threshold 预测为 naive_rag
        """
        super().__init__(config, tokenizer)
        self.strategy_names = config.model.strategy_names
        self.strategy_to_idx = {name: idx for idx, name in enumerate(self.strategy_names)}
        self.soft_label_threshold = soft_label_threshold
        self.raw_samples = []  # 存储原始数据
    
    def load_data(self, data_path: str):
        """
        加载软标签数据
        
        Args:
            data_path: 数据文件路径 (包含 soft_label 字段)
        """
        if not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")
        
        print(f"加载软标签路由数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        print(f"找到 {len(samples)} 条样本")
        
        self.raw_samples = samples
        self.data = []
        
        # 统计信息
        no_soft_label_count = 0
        
        for sample in samples:
            question = sample.get('question', '')
            soft_label = sample.get('soft_label', None)
            
            if not question:
                continue
            
            # 如果没有 soft_label，根据 em/f1 计算
            if soft_label is None:
                no_soft_label_count += 1
                # 尝试从 em/f1 计算
                no_rag_f1 = sample.get('no_rag_f1', 0.0)
                no_rag_em = sample.get('no_rag_em', 0.0)
                naive_rag_f1 = sample.get('naive_rag_f1', 0.0)
                naive_rag_em = sample.get('naive_rag_em', 0.0)
                
                # Q = 0.8 * F1 + 0.2 * EM
                Q_no_rag = 0.8 * no_rag_f1 + 0.2 * no_rag_em
                Q_naive_rag = 0.8 * naive_rag_f1 + 0.2 * naive_rag_em
                utility_gap = Q_naive_rag - Q_no_rag
                
                # sigmoid, temperature=0.1
                import math
                soft_label = 1.0 / (1.0 + math.exp(-utility_gap / 0.1))
            
            # 构建 TrainingItem (存储软标签在 strategy_scores 中)
            # 注意: 这里我们用一个特殊的方式存储软标签
            # scores[no_rag_idx] = 1 - soft_label
            # scores[naive_rag_idx] = soft_label
            no_rag_idx = self.strategy_to_idx.get('no_rag', 0)
            naive_rag_idx = self.strategy_to_idx.get('naive_rag', 1)
            
            strategy_scores = {}
            for strategy_name in self.strategy_names:
                if strategy_name == 'naive_rag':
                    strategy_scores[strategy_name] = {'score': soft_label}
                else:
                    strategy_scores[strategy_name] = {'score': 1.0 - soft_label}
            
            item = TrainingItem(
                question=question,
                strategy_scores=strategy_scores,
                cluster_id=0
            )
            
            self.data.append(item)
        
        print(f"成功加载 {len(self.data)} 条训练数据")
        
        if no_soft_label_count > 0:
            print(f"警告: {no_soft_label_count} 条样本缺少 soft_label 字段，已自动计算")
        
        # 打印软标签分布统计
        self._print_soft_label_stats()
    
    def _print_soft_label_stats(self):
        """打印软标签分布统计"""
        if not self.raw_samples:
            return
        
        soft_labels = [s.get('soft_label', 0.5) for s in self.raw_samples]
        
        near_0 = sum(1 for sl in soft_labels if sl < 0.3)
        near_05 = sum(1 for sl in soft_labels if 0.3 <= sl <= 0.7)
        near_1 = sum(1 for sl in soft_labels if sl > 0.7)
        
        print(f"\n软标签分布:")
        print(f"  < 0.3 (倾向 no_rag): {near_0} ({near_0/len(soft_labels)*100:.1f}%)")
        print(f"  0.3~0.7 (模糊): {near_05} ({near_05/len(soft_labels)*100:.1f}%)")
        print(f"  > 0.7 (倾向 naive_rag): {near_1} ({near_1/len(soft_labels)*100:.1f}%)")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项
        
        Returns:
            {
                'queries': str,           # 问题文本
                'soft_label': float,      # 软标签 (0~1)
                'scores': List[float],    # [1-soft_label, soft_label]
                'cluster_id': int,
            }
        """
        item = self.data[idx]
        raw_sample = self.raw_samples[idx] if idx < len(self.raw_samples) else {}
        
        # 获取软标签
        soft_label = raw_sample.get('soft_label', 0.5)
        if soft_label is None:
            # 从 item.strategy_scores 获取
            soft_label = item.strategy_scores.get('naive_rag', {}).get('score', 0.5)
        
        result = {
            'queries': item.question,
            'soft_label': soft_label,  # 核心字段
            'scores': [],              # 兼容现有 collate_fn
            'cluster_id': item.cluster_id,
        }
        
        # 构建 scores (兼容现有代码)
        for strategy_name in self.strategy_names:
            metrics = item.strategy_scores.get(strategy_name, {})
            result['scores'].append(metrics.get('score', 0.0))
        
        return result
    
    def __len__(self) -> int:
        return len(self.data)


# 注册到工厂
TrainableRouterFactory.register_dataset('soft_label')(SoftLabelRouterDataset)
