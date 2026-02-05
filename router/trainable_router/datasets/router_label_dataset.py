"""
路由标签数据集

直接加载router_test_labels.json格式
"""

import os
import json
from typing import List, Dict, Any

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig
from ..data_utils import TrainingItem


class RouterLabelDataset(BaseRouterDataset):
    """
    路由标签数据集

    直接加载router_test_labels.json格式
    格式: {
        "samples": [
            {
                "question": "...",
                "optimal_strategy": "no_rag" or "naive_rag",
                "no_rag_score": 0.5,
                "naive_rag_score": 0.3
            }
        ]
    }
    """

    def __init__(
        self,
        config: TrainableRouterConfig,
        tokenizer=None
    ):
        """
        初始化

        Args:
            config: 数据集配置
            tokenizer: 分词器
        """
        super().__init__(config, tokenizer)
        self.strategy_names = config.model.strategy_names
        self.strategy_to_idx = {name: idx for idx, name in enumerate(self.strategy_names)}

    def load_data(self, data_path: str):
        """
        加载数据

        Args:
            data_path: router_test_labels.json文件路径
        """
        if not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")

        print(f"加载路由标签数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = data.get('samples', [])
        print(f"找到 {len(samples)} 条样本")

        # 转换为训练格式
        self.data = []
        for sample in samples:
            question = sample.get('question', '')
            optimal_strategy = sample.get('optimal_strategy', '')

            if not question or not optimal_strategy:
                continue

            # 构建策略分数：最优策略=1.0，其他=0.0
            # 这样argmax就能直接得到最优策略
            strategy_scores = {}
            for strategy_name in self.strategy_names:
                if strategy_name == optimal_strategy:
                    strategy_scores[strategy_name] = {'score': 1.0, 'em': 1.0, 'f1': 1.0}
                else:
                    strategy_scores[strategy_name] = {'score': 0.0, 'em': 0.0, 'f1': 0.0}

            item = TrainingItem(
                question=question,
                strategy_scores=strategy_scores,
                cluster_id=0  # 验证集不需要聚类
            )

            self.data.append(item)

        print(f"成功加载 {len(self.data)} 条训练数据")

        # 打印标签分布
        label_counts = {}
        for item in self.data:
            # 找到最优策略（score=1.0）
            for strategy, metrics in item.strategy_scores.items():
                if metrics.get('score', 0) == 1.0:
                    label_counts[strategy] = label_counts.get(strategy, 0) + 1
                    break

        print(f"标签分布:")
        for strategy, count in label_counts.items():
            print(f"  {strategy}: {count} ({count/len(self.data):.2%})")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项

        Args:
            idx: 索引

        Returns:
            数据项字典
        """
        item = self.data[idx]

        result = {
            'queries': item.question,  # 改为'queries'，与collate_fn一致
            'scores': [],
            'cluster_id': item.cluster_id,
        }

        # 构建策略分数数组
        for strategy_name in self.strategy_names:
            metrics = item.strategy_scores.get(strategy_name, {})
            # 使用综合分数
            result['scores'].append(metrics.get('score', 0.0))

        return result

    def __len__(self) -> int:
        """数据集大小"""
        return len(self.data)
