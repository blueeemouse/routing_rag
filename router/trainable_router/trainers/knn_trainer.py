"""
KNNTrainer实现

KNN路由器训练器
由于KNN不需要梯度传播，训练过程主要是：
1. 编码所有训练样本
2. 存储embeddings和标签
3. 在验证集上评估
"""

import os
import json
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..base_trainer import BaseTrainer
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig

# TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class KNNTrainer(BaseTrainer):
    """KNN路由器训练器
    
    KNN不需要梯度传播，训练过程主要是：
    1. 编码所有训练样本并存储embeddings和标签
    2. 在验证集上评估性能
    3. 可选：搜索最优k值
    """
    
    def __init__(self, model, config: TrainableRouterConfig, output_dir: str = "outputs", logger=None):
        """
        初始化

        Args:
            model: KNNRouter模型
            config: 配置
            output_dir: 输出目录
            logger: (可选) 标准 logging.Logger 对象
        """
        super().__init__(model, config, output_dir, logger)

        self.training_config = config.training
        self.data_config = config.data

        # 验证数据加载器
        self.val_dataloader = None

        # TensorBoard 记录器
        self.tensorboard_writer = None
        if TENSORBOARD_AVAILABLE:
            tensorboard_dir = os.path.join(output_dir, "tensorboard")
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard 日志将保存到: {tensorboard_dir}")
        else:
            print("TensorBoard 未安装，跳过 TensorBoard 记录")

        # KNN特有配置
        self.search_k = getattr(self.training_config, 'search_k', False)  # 是否搜索最优k值
        self.k_candidates = getattr(self.training_config, 'k_candidates', [1, 3, 5, 7, 9, 11, 15, 21])
    
    def compute_loss(self, batch) -> torch.Tensor:
        """
        KNN不需要计算损失，这里返回一个dummy值
        
        Args:
            batch: 批次数据
            
        Returns:
            零张量
        """
        return torch.tensor(0.0, device=self.device)
    
    def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        "训练"一个epoch：编码所有训练样本并存储
        
        Args:
            dataloader: 数据加载器
            max_steps: 最大步数（用于调试）
            
        Returns:
            训练指标
        """
        # KNN的训练只有一次：编码所有样本
        # 如果已经拟合过，直接返回
        if self.model.is_fitted:
            print("KNN模型已经拟合，跳过训练")
            return {'loss': 0.0}
        
        # 收集所有训练数据
        all_queries = []
        all_labels = []
        
        print("正在收集训练数据...")
        for batch in tqdm(dataloader, desc="Loading data", ascii=True):
            queries = batch.get('queries', [])
            
            # 尝试多种方式获取标签
            labels = batch.get('label', batch.get('labels', None))
            
            # 如果没有直接的标签，从 scores 中推断
            if labels is None and 'scores' in batch:
                scores = batch['scores']
                if isinstance(scores, torch.Tensor):
                    labels = scores.argmax(dim=-1).tolist()
                else:
                    labels = [s.index(max(s)) for s in scores]
            
            if labels is None:
                continue
            
            all_queries.extend(queries)
            
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            elif isinstance(labels, list):
                all_labels.extend(labels)
            else:
                all_labels.append(labels)
        
        print(f"收集到 {len(all_queries)} 个训练样本")
        
        # 拟合模型：编码所有样本并存储
        self.model.fit(all_queries, all_labels)
        
        self.global_step = 1
        
        return {'loss': 0.0, 'num_samples': len(all_queries)}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型

        Args:
            dataloader: 数据加载器

        Returns:
            评估指标
        """
        if not self.model.is_fitted:
            print("警告：KNN模型尚未拟合，无法评估")
            return {'accuracy': 0.0, 'loss': 0.0}

        self.model.eval()

        all_queries = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Loading val data", ascii=True):
            queries = batch.get('queries', [])
            
            # 尝试多种方式获取标签
            labels = batch.get('label', batch.get('labels', None))
            
            # 如果没有直接的标签，从 scores 中推断
            if labels is None and 'scores' in batch:
                scores = batch['scores']
                if isinstance(scores, torch.Tensor):
                    labels = scores.argmax(dim=-1).tolist()
                else:
                    labels = [s.index(max(s)) for s in scores]
            
            if labels is None:
                continue
            
            all_queries.extend(queries)
            
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            elif isinstance(labels, list):
                all_labels.extend(labels)
            else:
                all_labels.append(labels)

        if len(all_queries) == 0:
            print("警告：验证数据为空")
            return {'accuracy': 0.0, 'loss': 0.0}

        # 预测
        predictions = self.model.route(all_queries)

        # 将预测结果转换为索引
        pred_indices = [self.model.strategy_names.index(p) if p in self.model.strategy_names else 0 for p in predictions]

        # 计算准确率
        correct = sum(p == l for p, l in zip(pred_indices, all_labels))
        accuracy = correct / len(all_labels)

        # 计算每个策略的准确率（召回率）
        strategy_accuracy = {}
        for i, strategy in enumerate(self.model.strategy_names):
            mask = [l == i for l in all_labels]
            if any(mask):
                strategy_correct = sum(p == l for p, l, m in zip(pred_indices, all_labels, mask) if m)
                strategy_total = sum(mask)
                strategy_accuracy[strategy] = strategy_correct / strategy_total if strategy_total > 0 else 0.0

        # 计算路由分布
        routing_distribution = {}
        for i, strategy in enumerate(self.model.strategy_names):
            count = pred_indices.count(i)
            routing_distribution[strategy] = count / len(pred_indices)

        # 计算真实标签分布
        label_distribution = {}
        for i, strategy in enumerate(self.model.strategy_names):
            count = all_labels.count(i)
            label_distribution[strategy] = count / len(all_labels)

        metrics = {
            'accuracy': accuracy,
            'loss': 0.0,
            'strategy_accuracy': strategy_accuracy,
            'routing_distribution': routing_distribution,
            'label_distribution': label_distribution,
            'num_samples': len(all_labels),
        }

        # 打印详细评估信息
        print(f"\n{'='*80}")
        print(f"评估结果 (总样本: {len(all_labels)})")
        print(f"{'='*80}")
        print(f"  整体准确率: {accuracy:.4f}")
        print(f"  当前k值: {self.model.k}")
        print(f"\n  真实标签分布:")
        for strategy, ratio in label_distribution.items():
            print(f"    {strategy}: {ratio:.2%}")
        print(f"\n  预测路由分布:")
        for strategy, ratio in routing_distribution.items():
            print(f"    {strategy}: {ratio:.2%}")
        print(f"\n  各策略召回率:")
        for strategy, acc in strategy_accuracy.items():
            print(f"    {strategy}: {acc:.4f}")
        print(f"{'='*80}\n")

        return metrics
    
    def search_best_k(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        搜索最优k值
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            搜索结果
        """
        if not self.model.is_fitted:
            print("错误：KNN模型尚未拟合，无法搜索k值")
            return {}
        
        original_k = self.model.k
        results = {}
        
        print(f"\n开始搜索最优k值，候选值: {self.k_candidates}")
        
        # 收集验证数据
        all_queries = []
        all_labels = []
        
        for batch in dataloader:
            queries = batch.get('queries', [])
            labels = batch.get('label', batch.get('labels', None))
            
            if labels is None:
                continue
            
            all_queries.extend(queries)
            
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(labels)
        
        # 测试每个k值
        best_k = original_k
        best_accuracy = 0.0
        
        for k in self.k_candidates:
            self.model.k = k
            predictions = self.model.route(all_queries)
            pred_indices = [self.model.strategy_names.index(p) for p in predictions]
            
            correct = sum(p == l for p, l in zip(pred_indices, all_labels))
            accuracy = correct / len(all_labels)
            
            results[k] = accuracy
            print(f"  k={k}: accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        
        # 恢复最优k值
        self.model.k = best_k
        print(f"\n最优k值: {best_k} (accuracy={best_accuracy:.4f})")
        
        return {
            'original_k': original_k,
            'best_k': best_k,
            'best_accuracy': best_accuracy,
            'all_results': results,
        }
    
    def train(self, train_dataloader, val_dataloader=None, **kwargs) -> Dict[str, Any]:
        """
        训练主循环
        
        对于KNN，训练过程是：
        1. 编码所有训练样本并存储
        2. 如果有验证集，进行评估
        3. 可选：搜索最优k值
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            
        Returns:
            训练历史
        """
        history = {
            'train_loss': [],
            'val_metrics': [],
        }
        
        # 保存验证数据加载器
        self.val_dataloader = val_dataloader
        
        # 训练（编码所有样本）
        train_metrics = self.train_epoch(train_dataloader)
        history['train_loss'].append(train_metrics.get('loss', 0.0))
        
        # 验证
        if val_dataloader is not None:
            # 可选：搜索最优k值
            if self.search_k:
                search_results = self.search_best_k(val_dataloader)
                print(f"K值搜索结果: {search_results}")
            
            # 最终评估
            val_metrics = self.evaluate(val_dataloader)
            history['val_metrics'].append(val_metrics)
            
            # TensorBoard 记录
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('Accuracy/val', val_metrics.get('accuracy', 0), 0)
                for strategy, acc in val_metrics.get('strategy_accuracy', {}).items():
                    self.tensorboard_writer.add_scalar(f'Accuracy/val_{strategy}', acc, 0)
        
        return history
    
    def save_checkpoint(self, path: str):
        """
        保存检查点
        
        Args:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存模型
        self.model.save(path)
        
        # 保存训练状态
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        with open(os.path.join(path, 'train_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"检查点已保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        self.model.load(path)
        
        state_path = os.path.join(path, 'train_state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state.get('global_step', 0)
            self.epoch = state.get('epoch', 0)
    
    def save_final_model(self, path: str):
        """
        保存最终模型
        
        Args:
            path: 保存路径
        """
        self.model.save(path)

        # 保存训练配置
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            json.dump({
                'training_steps': self.global_step,
                'epochs': self.epoch + 1,
                'k': self.model.k,
                'distance_metric': self.model.distance_metric,
                'weighted_voting': self.model.weighted_voting,
            }, f, indent=2)

    def close(self):
        """关闭资源"""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            print("TensorBoard writer 已关闭")


# 注册到工厂
TrainableRouterFactory.register_trainer('knn')(KNNTrainer)
