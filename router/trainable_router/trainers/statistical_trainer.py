"""
StatisticalTrainer实现

专门用于训练纯统计特征路由器
不依赖语义embedding，直接使用手工特征进行二分类
"""

import os
import json
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..base_trainer import BaseTrainer
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig


class StatisticalTrainer(BaseTrainer):
    """
    统计特征路由器训练器
    
    特点：
    - 直接接收query文本，在forward中提取统计特征
    - 使用BCEWithLogitsLoss进行二分类
    - 支持阈值优化
    """
    
    def __init__(self, model, config: TrainableRouterConfig, output_dir: str = "outputs", logger=None):
        """
        初始化
        
        Args:
            model: StatisticalRouterModel
            config: 配置
            output_dir: 输出目录
            logger: 日志记录器
        """
        super().__init__(model, config, output_dir, logger)
        
        self.training_config = config.training
        self.data_config = config.data
        
        # 验证数据加载器
        self.val_dataloader = None
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 学习率调度器
        self.scheduler = self._init_scheduler()
        
        # 类别权重（处理不平衡数据）
        self.class_weights = getattr(self.training_config, 'class_weights', None)
        
        # 跟踪最佳性能
        self.best_val_accuracy = float('-inf')
        self.best_val_step = 0
        
        print(f"StatisticalTrainer 初始化完成")
    
    def _init_optimizer(self):
        """初始化优化器"""
        lr = self.training_config.learning_rate
        
        # 从 optimizer_kwargs 获取 weight_decay，默认 0.0
        weight_decay = self.training_config.optimizer_kwargs.get('weight_decay', 0.0) if self.training_config.optimizer_kwargs else 0.0
        
        if self.training_config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def _init_scheduler(self):
        """初始化学习率调度器"""
        total_steps = self.training_config.training_steps
        
        if total_steps <= 0:
            return None
        
        if self.training_config.scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            num_warmup_steps = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps
            )
            return scheduler
        elif self.training_config.scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            num_warmup_steps = int(0.1 * total_steps)
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps
            )
            return scheduler
        else:
            return None
    
    def compute_loss(self, batch) -> torch.Tensor:
        """
        计算二分类损失
        
        Args:
            batch: 批次数据
                - queries: query文本列表
                - scores: 策略分数, shape: (batch_size, num_strategies)
                
        Returns:
            损失值
        """
        queries = batch.get('queries', [])
        
        if not queries:
            raise ValueError("Batch中缺少'queries'字段")
        
        # 前向传播 - StatisticalRouterModel直接接收queries
        # 内部会调用特征提取器
        logits = self.model.forward(queries)  # (batch_size, 1)
        
        # 获取标签
        scores = batch['scores'].to(self.device)
        labels = scores.argmax(dim=-1).float()  # (batch_size,) - 0或1
        
        # BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.squeeze(-1), labels)
        
        # 记录logits范围
        if self.global_step % self.training_config.eval_steps == 0:
            loss_str = f"Step {self.global_step}: Loss = {loss:.6f}, Logits = [{logits.min():.4f}, {logits.max():.4f}]"
            if self.logger:
                self.logger.info(loss_str)
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            max_steps: 最大步数限制
            
        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        batch_generator = self._get_training_batches(dataloader)
        pbar = tqdm(batch_generator, desc=f"Epoch {self.epoch + 1}", ascii=True)
        
        for batch in pbar:
            if max_steps is not None and self.global_step >= max_steps:
                break
            
            # 计算损失
            loss = self.compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.training_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.global_step += 1
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'step': self.global_step
            })
            
            # 定期评估
            if self.global_step % self.training_config.eval_steps == 0:
                if self.val_dataloader is not None:
                    val_metrics = self.evaluate(self.val_dataloader)
                    val_acc = val_metrics.get('accuracy', 0)
                    val_loss = val_metrics.get('loss', 0)
                    
                    # 记录详细评估结果到日志
                    if self.logger:
                        self.logger.info(f"评估 (Step {self.global_step}): Acc={val_acc:.4f}, Loss={val_loss:.4f}")
                        
                        # 记录预测路由分布
                        routing_dist = val_metrics.get('routing_distribution', {})
                        self.logger.info("预测路由分布:")
                        for strategy, ratio in sorted(routing_dist.items()):
                            self.logger.info(f"    {strategy}: {ratio*100:.2f}%")
                        
                        # 记录各策略召回率
                        strategy_acc = val_metrics.get('strategy_accuracy', {})
                        self.logger.info("各策略召回率:")
                        for strategy, acc in sorted(strategy_acc.items()):
                            self.logger.info(f"    {strategy}: {acc:.4f}")
                        
                        self.logger.info("")  # 空行分隔
                    
                    # 保存最佳模型
                    if val_acc > self.best_val_accuracy:
                        self.best_val_accuracy = val_acc
                        self.best_val_step = self.global_step
                        
                        checkpoint_path = f"{self.output_dir}/checkpoint_best_val"
                        self.save_checkpoint(checkpoint_path)
                        
                        if self.logger:
                            self.logger.info(f"🏆 新的最佳Val性能！Step={self.global_step}, Acc={val_acc:.4f}")
                            self.logger.info("")  # 空行分隔
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ascii=True):
                queries = batch.get('queries', [])
                scores = batch['scores'].to(self.device)
                
                # 前向传播
                logits = self.model.forward(queries)
                
                # 获取标签
                labels = scores.argmax(dim=-1).float()
                
                # 计算损失
                loss = loss_fn(logits.squeeze(-1), labels)
                total_loss += loss.item()
                num_batches += 1
                
                # 预测
                probs = torch.sigmoid(logits.squeeze(-1))
                predictions = (probs >= self.model.threshold).long()
                
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().long().tolist())
        
        # 计算准确率
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0
        
        # 计算各类别的准确率
        strategy_accuracy = {}
        for i, strategy in enumerate(self.model.strategy_names):
            mask = [l == i for l in all_labels]
            if any(mask):
                strategy_correct = sum(p == l for p, l, m in zip(all_predictions, all_labels, mask) if m)
                strategy_total = sum(mask)
                strategy_accuracy[strategy] = strategy_correct / strategy_total if strategy_total > 0 else 0.0
        
        # 计算路由分布
        routing_distribution = {}
        for i, strategy in enumerate(self.model.strategy_names):
            count = all_predictions.count(i)
            routing_distribution[strategy] = count / len(all_predictions) if len(all_predictions) > 0 else 0.0
        
        # 计算标签分布
        label_distribution = {}
        for i, strategy in enumerate(self.model.strategy_names):
            count = all_labels.count(i)
            label_distribution[strategy] = count / len(all_labels) if len(all_labels) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'strategy_accuracy': strategy_accuracy,
            'routing_distribution': routing_distribution,
            'label_distribution': label_distribution,
            'num_samples': len(all_labels),
        }
        
        # 打印评估结果
        print(f"\n{'='*80}")
        print(f"评估结果 (总样本: {len(all_labels)})")
        print(f"{'='*80}")
        print(f"  整体准确率: {accuracy:.4f}")
        print(f"  平均损失: {metrics['loss']:.4f}")
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
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(path, 'model.pt'))
        
        # 保存训练状态
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        with open(os.path.join(path, 'train_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        # 保存模型配置
        model_config = {
            'strategy_names': self.model.strategy_names,
            'num_strategies': self.model.num_strategies,
            'threshold': self.model.threshold,
            'mlp_hidden_dims': self.model.mlp_hidden_dims,
        }
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        print(f"检查点已保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
    
    def save_final_model(self, path: str):
        """保存最终模型"""
        self.model.save(path)
        
        # 保存训练配置
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            training_info = {
                'training_steps': self.global_step,
                'epochs': self.epoch + 1,
                'learning_rate': self.training_config.learning_rate,
                'best_val_accuracy': round(self.best_val_accuracy, 4),
                'best_val_step': self.best_val_step,
            }
            json.dump(training_info, f, indent=2)
    
    def optimize_threshold(self, dataloader: DataLoader) -> float:
        """
        在验证集上优化阈值
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            最优阈值
        """
        self.model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                queries = batch.get('queries', [])
                scores = batch['scores']
                
                logits = self.model.forward(queries)
                probs = torch.sigmoid(logits.squeeze(-1))
                
                labels = scores.argmax(dim=-1).numpy()
                
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.tolist())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        best_threshold = 0.5
        best_score = 0
        
        # 网格搜索
        for threshold in np.arange(0.1, 0.9, 0.05):
            predictions = (all_probs >= threshold).astype(int)
            accuracy = (predictions == all_labels).mean()
            
            if accuracy > best_score:
                best_score = accuracy
                best_threshold = threshold
        
        self.model.threshold = best_threshold
        return best_threshold


# 注册到工厂
TrainableRouterFactory.register_trainer('statistical')(StatisticalTrainer)
