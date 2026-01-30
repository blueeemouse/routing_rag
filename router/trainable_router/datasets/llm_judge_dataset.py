"""
LLM-as-a-Judge路由器数据集

从LLM评判结果生成训练数据
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig


class LLMJudgeRouterDataset(BaseRouterDataset):
    """LLM-as-a-Judge路由器数据集"""
    
    def __init__(self, config: TrainableRouterConfig, tokenizer=None):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器
        """
        super().__init__(config, tokenizer)
        self.strategy_names = config.model.strategy_names
        self.num_clusters = config.data.num_clusters
    
    def load_data(self, data_path: str):
        """
        加载LLM评判数据
        
        Args:
            data_path: 数据路径
        """
        if not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为训练数据格式
        self.data = self._convert_to_train_format(data)
        
        print(f"加载了 {len(self.data)} 条训练数据")
    
    def _convert_to_train_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        转换为训练数据格式
        
        Expected format:
        [
            {
                "question": "...",
                "llm_judgments": {
                    "no_rag": {"score": 0.8, "reason": "..."},
                    "naive_rag": {"score": 0.9, "reason": "..."},
                    "graph_rag": {"score": 0.7, "reason": "..."}
                }
            },
            ...
        ]
        
        Args:
            data: 原始数据
            
        Returns:
            训练数据列表
        """
        train_data = []
        
        for item in data:
            question = item.get('question', '')
            if not question:
                continue
            
            judgments = item.get('llm_judgments', {})
            scores = {}
            
            for strategy in self.strategy_names:
                if strategy in judgments:
                    # 使用LLM评判的分数
                    scores[strategy] = judgments[strategy].get('score', 0.0)
                else:
                    scores[strategy] = 0.0
            
            # 归一化分数
            if self.config.data.normalize_scores:
                max_score = max(scores.values()) if max(scores.values()) > 0 else 1.0
                scores = {k: v / max_score for k, v in scores.items()}
            
            train_data.append({
                'question': question,
                'scores': scores,
                'cluster_id': -1,
            })
        
        return train_data
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项字典
        """
        item = self.data[idx]
        
        result = {
            'question': item['question'],
            'queries': item['question'],
            'scores': [item['scores'][s] for s in self.strategy_names],
            'cluster_id': item['cluster_id'],
        }
        
        if self.tokenizer is not None:
            tokenized = self.tokenize(item['question'])
            result['input_ids'] = tokenized['input_ids'].squeeze(0)
            result['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        
        return result
    
    def __len__(self) -> int:
        """获取数据集大小"""
        return len(self.data)


# 注册到工厂
TrainableRouterFactory.register_dataset('llm_judge')(LLMJudgeRouterDataset)
