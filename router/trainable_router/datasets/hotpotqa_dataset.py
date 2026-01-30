"""
统一路由器数据集

支持多种数据格式和评分方式
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from sklearn.cluster import KMeans

from ..base_dataset import BaseRouterDataset
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig, DataConfig
from ..data_utils import DataAdapter, TrainingItem, ScoreComputer


class GenericRouterDataset(BaseRouterDataset):
    """
    统一路由器数据集
    
    特点：
    - 统一内部格式，支持多种输入格式
    - 可配置的评分方式
    - 自动聚类
    """
    
    def __init__(
        self, 
        config: TrainableRouterConfig, 
        tokenizer = None,
        score_formula: Optional[str] = None,
        data_adapter: Optional[DataAdapter] = None
    ):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器
            score_formula: 分数计算公式（可选，默认从配置中读取）
            data_adapter: 数据适配器（可选）
        """
        super().__init__(config, tokenizer)
        
        self.strategy_names = config.model.strategy_names
        self.num_clusters = config.data.num_clusters
        
        # 从配置中读取 score_formula，如果没有传入的话
        effective_score_formula = score_formula or config.data.score_formula
        self.data_adapter = data_adapter or DataAdapter(effective_score_formula)
        self.score_computer = ScoreComputer(effective_score_formula)
        
        # 数据存储
        self.data: List[TrainingItem] = []
        self.question_embeddings: Optional[np.ndarray] = None
    
    def load_data(self, data_path: str):
        """
        加载数据
        
        支持：
        - 单个文件: 直接加载
        - 目录: 批量加载目录下所有策略文件
        
        Args:
            data_path: 数据路径
        """
        if not os.path.exists(data_path):
            raise ValueError(f"数据路径不存在: {data_path}")
        
        if os.path.isfile(data_path):
            # 单个文件
            self.data = self._load_single_file(data_path)
        elif os.path.isdir(data_path):
            # 目录
            self.data = self._load_directory(data_path)
        else:
            raise ValueError(f"无效的数据路径: {data_path}")
        
        # 过滤和聚合
        self._post_process()
        
        print(f"加载了 {len(self.data)} 条训练数据")
        print(f"策略覆盖: {self._get_coverage_stats()}")
    
    def _load_single_file(self, file_path: str) -> List[TrainingItem]:
        """加载单个文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检测文件类型
        if isinstance(data, dict) and 'models' in data:
            # 聚合格式（comparison_results）
            return self._load_comparison_format(data)
        elif isinstance(data, dict) and 'model_name' in data:
            # 单策略格式
            adapter = self.data_adapter
            items = adapter.from_single_strategy(file_path)
            return adapter.aggregate(items)
        else:
            # 其他格式
            adapter = self.data_adapter
            return adapter.from_single_strategy(file_path)
    
    def _load_directory(self, dir_path: str) -> List[TrainingItem]:
        """加载目录下的所有策略文件"""
        adapter = self.data_adapter
        items = adapter.from_directory(dir_path, self.strategy_names)
        
        # 聚合
        aggregated = adapter.aggregate(items)
        
        # 过滤
        filtered = adapter.filter_by_coverage(aggregated, min_strategies=len(self.strategy_names))
        
        return filtered
    
    def _load_comparison_format(self, data: Dict[str, Any]) -> List[TrainingItem]:
        """加载comparison_results格式"""
        items = []
        
        for model_info in data.get('models', []):
            model_name = model_info.get('model_name', '')
            predictions = model_info.get('predictions', [])
            
            for pred in predictions:
                question = pred.get('question', '')
                if not question:
                    continue
                
                # 匹配策略
                strategy = self.data_adapter._match_strategy(model_name)
                if not strategy:
                    continue
                
                metrics = {
                    'em': pred.get('em', 0.0),
                    'f1': pred.get('f1', 0.0),
                }
                
                # 检查是否已存在
                existing = next((i for i in items if i.question == question), None)
                
                if existing:
                    if strategy not in existing.strategy_scores:
                        existing.strategy_scores[strategy] = {}
                    existing.strategy_scores[strategy].update(metrics)
                else:
                    items.append(TrainingItem(
                        question=question,
                        strategy_scores={strategy: metrics}
                    ))
        
        return items
    
    def _post_process(self):
        """后处理：过滤、聚合、归一化"""
        # 过滤策略覆盖不足的样本
        min_strategies = len(self.strategy_names) - 1  # 允许缺失1个策略
        self.data = [
            item for item in self.data 
            if len(item.strategy_scores) >= min_strategies
        ]
        
        # 归一化分数
        if self.config.data.normalize_scores:
            self.data = self.data_adapter.normalize_scores(self.data)
        
        # 添加cluster
        if self.num_clusters > 0:
            self._add_clusters()
    
    def _get_coverage_stats(self) -> Dict[str, int]:
        """获取策略覆盖统计"""
        stats = {s: 0 for s in self.strategy_names}
        
        for item in self.data:
            for strategy in item.strategy_scores:
                if strategy in stats:
                    stats[strategy] += 1
        
        return stats
    
    def _add_clusters(self):
        """使用K-Means为数据添加cluster id"""
        if len(self.data) < self.num_clusters:
            self.num_clusters = min(len(self.data), self.num_clusters)
        
        # 获取question embedding
        questions = [item.question for item in self.data]
        
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = encoder.encode(questions, convert_to_numpy=True)
            self.question_embeddings = embeddings
            
            # K-Means聚类
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(embeddings)
            
            for i, item in enumerate(self.data):
                item.cluster_id = cluster_ids[i]
            
            print(f"已添加 {self.num_clusters} 个cluster")
            
        except ImportError:
            import random
            for item in self.data:
                item.cluster_id = random.randint(0, self.num_clusters - 1)
            print("警告: 未安装sentence-transformers，使用随机cluster分配")
    
    def set_score_formula(self, formula: str):
        """
        设置评分公式
        
        Args:
            formula: 公式字符串
        """
        self.score_computer = ScoreComputer(formula)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项字典
        """
        item = self.data[idx]
        
        # 计算各策略的分数
        strategy_scores = item.strategy_scores
        computed_scores = {
            s: self.score_computer.compute(strategy_scores.get(s, {}))
            for s in self.strategy_names
        }
        
        result = {
            'question': item.question,
            'queries': item.question,
            'scores': [computed_scores.get(s, 0.0) for s in self.strategy_names],
            'raw_scores': {
                s: strategy_scores.get(s, {}) for s in self.strategy_names
            },
            'cluster_id': item.cluster_id,
            'question_idx': idx,
        }
        
        # 分词
        if self.tokenizer is not None:
            tokenized = self.tokenize(item.question)
            result['input_ids'] = tokenized['input_ids'].squeeze(0)
            result['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        
        return result
    
    def __len__(self) -> int:
        """获取数据集大小"""
        return len(self.data)
    
    def get_questions(self) -> List[str]:
        """获取所有问题"""
        return [item.question for item in self.data]
    
    def get_question_embeddings(self) -> Optional[np.ndarray]:
        """获取问题embedding"""
        return self.question_embeddings
    
    def get_raw_strategy_scores(self) -> Dict[str, List[Dict[str, float]]]:
        """获取原始策略分数"""
        scores = {s: [] for s in self.strategy_names}
        
        for item in self.data:
            for strategy, metrics in item.strategy_scores.items():
                if strategy in scores:
                    scores[strategy].append(metrics)
        
        return scores
    
    def split(
        self, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1, 
        seed: int = 42
    ):
        """
        划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        import random
        random.seed(seed)
        
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        def create_subset(indices_list):
            subset = GenericRouterDataset(
                self.config, 
                self.tokenizer,
                self.score_computer.formula,
                self.data_adapter
            )
            subset.data = [self.data[i] for i in indices_list]
            return subset
        
        return (
            create_subset(train_indices),
            create_subset(val_indices),
            create_subset(test_indices),
        )


class RouterDataLoader:
    """
    路由器数据加载器（简化版）
    
    快速加载和迭代数据，无需完整Dataset类
    """
    
    def __init__(
        self, 
        data_path: str,
        strategy_names: List[str] = None,
        score_formula: str = "em",
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        初始化
        
        Args:
            data_path: 数据路径
            strategy_names: 策略名称列表
            score_formula: 评分公式
            batch_size: 批量大小
            shuffle: 是否打乱
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        adapter = DataAdapter(score_formula)
        
        if os.path.isfile(data_path):
            items = adapter.from_single_strategy(data_path)
        elif os.path.isdir(data_path):
            items = adapter.from_directory(data_path)
            items = adapter.aggregate(items)
            items = adapter.filter_by_coverage(items, min_strategies=3)
        else:
            raise ValueError(f"无效路径: {data_path}")
        
        self.items = items
        self.strategy_names = strategy_names or adapter.STRATEGY_PATTERNS.keys()
        self.score_computer = ScoreComputer(score_formula)
        
        self.indices = list(range(len(self.items)))
        if shuffle:
            import random
            random.shuffle(self.indices)
    
    def __iter__(self):
        """迭代器"""
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield self._get_batch(batch_indices)
    
    def __len__(self):
        """批量数量"""
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    def _get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """获取批量数据"""
        batch_items = [self.items[idx] for idx in indices]
        
        questions = [item.question for item in batch_items]
        scores = []
        
        for item in batch_items:
            strategy_scores = {
                s: self.score_computer.compute(item.strategy_scores.get(s, {}))
                for s in self.strategy_names
            }
            scores.append(strategy_scores)
        
        return {
            'questions': questions,
            'scores': scores,
            'raw_items': batch_items,
        }
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有数据"""
        return self._get_batch(self.indices)


# 注册到工厂
TrainableRouterFactory.register_dataset('generic')(GenericRouterDataset)
TrainableRouterFactory.register_dataset('hotpotqa')(GenericRouterDataset)
TrainableRouterFactory.register_dataset('llm_judge')(GenericRouterDataset)