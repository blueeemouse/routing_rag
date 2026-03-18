"""
决策式Router数据集

【重要】专用于DecisionRouter的数据集，仅支持统一的列表格式

数据格式要求：
{
    "samples": [
        {
            "question": "...",
            "Q": [Q_no_rag, Q_naive_rag, ...],      // 各策略性能分数列表
            "costs": [cost_no_rag, cost_naive_rag, ...],  // 各策略归一化成本列表
            "optimal_strategy": "naive_rag"  // 最优策略名称
        }
    ],
    "metadata": {
        "strategy_names": ["no_rag", "naive_rag"],
        ...
    }
}

配合使用：
- DecisionRouterModel（预测Q和cost）
- DecisionRouterTrainer（回归损失训练）

数据生成：
    python scripts/generate_decision_router_data.py
"""

import os
import json
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig


class DecisionRouterDataset(BaseRouterDataset):
    """
    决策式Router数据集
    
    【特点】
    - 统一的列表格式：Q和costs都是与strategy_names顺序一致的列表
    - 简单直接：移除复杂的兼容逻辑
    - 支持多分类：轻松扩展到更多策略
    
    返回字段：
    - input_ids: token IDs (用于语义编码器)
    - attention_mask: 注意力掩码
    - queries: 原始问题文本
    - Q: 各策略性能分数列表
    - costs: 各策略归一化成本列表
    - label: 最优策略索引
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
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__(config, tokenizer)
        
        self.strategy_names = config.model.strategy_names
        self.num_strategies = config.model.num_strategies
        self.strategy_to_idx = {name: idx for idx, name in enumerate(self.strategy_names)}
        self.max_length = max_length
        
        # 如果没有提供tokenizer，从配置加载
        if self.tokenizer is None and hasattr(config.model, 'backbone_name'):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_name)
                print(f"DecisionRouterDataset 加载 tokenizer: {config.model.backbone_name}")
            except Exception as e:
                print(f"警告: 无法加载 tokenizer: {e}")
    
    def load_data(self, data_path: str):
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
        """
        if not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")
        
        print(f"加载决策路由数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        metadata = data.get('metadata', {})
        
        print(f"找到 {len(samples)} 条样本")
        
        # 验证strategy_names一致性
        data_strategy_names = metadata.get('strategy_names', [])
        if data_strategy_names and data_strategy_names != self.strategy_names:
            print(f"警告: 数据中的strategy_names {data_strategy_names} 与配置 {self.strategy_names} 不一致")
        
        self.samples = samples
        self._print_stats()
    
    def _print_stats(self):
        """打印统计信息"""
        if not self.samples:
            return
        
        # 统计策略分布
        strategy_counts = {}
        for s in self.samples:
            st = s.get('optimal_strategy', 'unknown')
            strategy_counts[st] = strategy_counts.get(st, 0) + 1
        
        print(f"\n最优策略分布:")
        for st, cnt in sorted(strategy_counts.items()):
            print(f"  {st}: {cnt} ({cnt/len(self.samples)*100:.1f}%)")
        
        # 统计Q分布
        print(f"\nQ分数分布:")
        for i, strategy in enumerate(self.strategy_names):
            q_values = [s['Q'][i] for s in self.samples if len(s.get('Q', [])) > i]
            if q_values:
                mean_q = sum(q_values) / len(q_values)
                print(f"  {strategy}: mean={mean_q:.4f}, samples={len(q_values)}")
        
        # 统计cost分布
        print(f"\nCost分布:")
        for i, strategy in enumerate(self.strategy_names):
            costs = [s['costs'][i] for s in self.samples if len(s.get('costs', [])) > i]
            if costs:
                mean_cost = sum(costs) / len(costs)
                min_cost = min(costs)
                max_cost = max(costs)
                print(f"  {strategy}: mean={mean_cost:.4f}, min={min_cost:.4f}, max={max_cost:.4f}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项
        
        Returns:
            {
                'input_ids': torch.Tensor,      # (max_length,)
                'attention_mask': torch.Tensor, # (max_length,)
                'queries': str,                 # 原始问题文本
                'Q': List[float],               # 各策略性能分数
                'costs': List[float],           # 各策略归一化成本
                'label': int,                   # 最优策略索引
                'cluster_id': int,              # 兼容字段
            }
        """
        sample = self.samples[idx]
        question = sample.get('question', '')
        
        # Tokenize
        input_ids = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                question,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 获取Q和costs
        Q = sample.get('Q', [0.0] * self.num_strategies)
        costs = sample.get('costs', [0.0] * self.num_strategies)
        
        # 确保长度正确
        if len(Q) < self.num_strategies:
            Q = Q + [0.0] * (self.num_strategies - len(Q))
        if len(costs) < self.num_strategies:
            costs = costs + [0.0] * (self.num_strategies - len(costs))
        
        # 获取标签
        label_str = sample.get('optimal_strategy', '')
        label = self.strategy_to_idx.get(label_str, 0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'queries': question,
            'Q': Q,
            'costs': costs,
            'label': label,
            'cluster_id': idx,  # 简单使用索引
        }
    
    def __len__(self) -> int:
        return len(self.samples)


# 注册到工厂
TrainableRouterFactory.register_dataset('decision_router')(DecisionRouterDataset)