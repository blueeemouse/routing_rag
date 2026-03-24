"""
混合表征融合路由器训练器

用于训练 HybridRepresentationFusionModel
"""

import os
import json
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..base_trainer import BaseTrainer
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig


class HybridRepresentationFusionTrainer(BaseTrainer):
    """
    混合表征融合路由器训练器
    
    特点：
    - 输入为预提取的 LLM 内部表征向量 + tokenized query
    - 使用 Cross Attention 融合两种表征
    - 使用交叉熵损失进行多分类
    - 支持学习率调度器
    - 支持类别权重处理不平衡
    - 定期评估和保存最佳模型
    """
    
    def __init__(
        self,
        model,
        config: TrainableRouterConfig,
        output_dir: str = "outputs",
        logger=None
    ):
        """
        初始化
        
        Args:
            model: HybridRepresentationFusionModel 模型
            config: 配置
            output_dir: 输出目录
            logger: 日志记录器
        """
        super().__init__(model, config, output_dir, logger)
        
        self.training_config = config.training
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 学习率调度器
        self.scheduler = None
        
        # 跟踪最佳性能
        self.best_val_accuracy = 0.0
        self.best_val_step = 0
        
        # 类别权重
        self.class_weights = None
        if hasattr(self.training_config, 'class_weights') and self.training_config.class_weights:
            weights = []
            for name in self.model.strategy_names:
                w = self.training_config.class_weights.get(name, 1.0)
                weights.append(w)
            self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            print(f"使用类别权重: {dict(zip(self.model.strategy_names, weights))}")
        
        print(f"HybridRepresentationFusionTrainer 初始化完成")
        print(f"  - 学习率: {self.training_config.learning_rate}")
        print(f"  - 批量大小: {self.training_config.batch_size}")
        print(f"  - 训练轮数: {self.training_config.epochs}")
        
        # 统计可训练参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 冻结参数: {total_params - trainable_params:,}")
    
    def _init_optimizer(self):
        """初始化优化器"""
        lr = self.training_config.learning_rate
        weight_decay = 0.01
        
        if self.training_config.optimizer_kwargs:
            weight_decay = self.training_config.optimizer_kwargs.get('weight_decay', 0.01)
        
        # 分组参数：backbone 和其他参数使用不同的学习率
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': lr},
            {'params': backbone_params, 'lr': lr * 0.1},  # backbone 使用较小学习率
        ]
        
        # 如果 backbone 全部冻结，移除空组
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay
        )
        
        return optimizer
    
    def _init_scheduler(self, num_training_steps: int):
        """初始化学习率调度器"""
        scheduler_type = getattr(self.training_config, 'scheduler_type', 'linear')
        
        if scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            num_warmup_steps = int(0.1 * num_training_steps)
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            num_warmup_steps = int(0.1 * num_training_steps)
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            return None
    
    def compute_loss(self, batch) -> torch.Tensor:
        """
        计算损失
        
        Args:
            batch: 批次数据
                - representation: 内部表征, shape: (batch_size, representation_dim)
                - input_ids: token IDs, shape: (batch_size, seq_len)
                - attention_mask: 注意力掩码, shape: (batch_size, seq_len)
                - label: 标签索引, shape: (batch_size,)
                
        Returns:
            损失值
        """
        representation = batch['representation'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # 前向传播
        logits = self.model.forward(representation, input_ids, attention_mask)
        
        # 计算损失
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        训练一个 epoch
        
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
                    
                    if self.logger:
                        self.logger.info(f"评估 (Step {self.global_step}): Acc={val_acc:.4f}, Loss={val_metrics.get('loss', 0):.4f}")
                        
                        # 记录路由分布
                        routing_dist = val_metrics.get('routing_distribution', {})
                        self.logger.info("预测路由分布:")
                        for strategy, ratio in sorted(routing_dist.items()):
                            self.logger.info(f"    {strategy}: {ratio*100:.2f}%")
                        
                        # 记录各策略召回率
                        strategy_acc = val_metrics.get('strategy_accuracy', {})
                        self.logger.info("各策略召回率:")
                        for strategy, acc in sorted(strategy_acc.items()):
                            self.logger.info(f"    {strategy}: {acc:.4f}")
                        
                        self.logger.info("")
                    
                    # 保存最佳模型
                    if val_acc > self.best_val_accuracy:
                        self.best_val_accuracy = val_acc
                        self.best_val_step = self.global_step
                        
                        checkpoint_path = f"{self.output_dir}/checkpoint_best_val"
                        self.save_checkpoint(checkpoint_path)
                        
                        print(f" 新的最佳验证准确率: {val_acc:.4f} (Step {self.global_step})")
        
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
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ascii=True):
                representation = batch['representation'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                logits = self.model.forward(representation, input_ids, attention_mask)
                
                # 计算损失
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # 预测
                predictions = logits.argmax(dim=-1)
                
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # 计算指标
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0
        
        # 各策略准确率（召回率）
        strategy_accuracy = {}
        for i, strategy in enumerate(self.model.strategy_names):
            mask = [l == i for l in all_labels]
            if any(mask):
                strategy_correct = sum(p == l for p, l, m in zip(all_predictions, all_labels, mask) if m)
                strategy_total = sum(mask)
                strategy_accuracy[strategy] = strategy_correct / strategy_total if strategy_total > 0 else 0.0
        
        # 预测分布
        routing_distribution = {}
        for i, strategy in enumerate(self.model.strategy_names):
            count = all_predictions.count(i)
            routing_distribution[strategy] = count / len(all_predictions) if len(all_predictions) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'strategy_accuracy': strategy_accuracy,
            'routing_distribution': routing_distribution,
            'num_samples': len(all_labels),
        }
        
        # 打印评估结果
        print(f"\n{'='*60}")
        print(f"评估结果 (样本数: {len(all_labels)})")
        print(f"{'='*60}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  平均损失: {metrics['loss']:.4f}")
        print(f"\n  各策略召回率:")
        for strategy, acc in strategy_accuracy.items():
            print(f"    {strategy}: {acc:.4f}")
        print(f"\n  预测分布:")
        for strategy, ratio in routing_distribution.items():
            print(f"    {strategy}: {ratio:.2%}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def train(self, train_dataloader, val_dataloader=None, **kwargs) -> Dict[str, Any]:
        """
        训练主循环
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            
        Returns:
            训练历史
        """
        # 保存验证数据加载器
        self.val_dataloader = val_dataloader
        
        # 初始化学习率调度器
        num_training_steps = len(train_dataloader) * self.training_config.epochs
        self.scheduler = self._init_scheduler(num_training_steps)
        
        print(f"训练配置:")
        print(f"  - 训练样本数: {len(train_dataloader.dataset)}")
        if val_dataloader:
            print(f"  - 验证样本数: {len(val_dataloader.dataset)}")
        print(f"  - 总训练步数: {num_training_steps}")
        print(f"  - 评估间隔: {self.training_config.eval_steps}")
        
        return super().train(train_dataloader, val_dataloader, **kwargs)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_step': self.best_val_step,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(path, 'model.pt'))
        
        # 保存训练状态
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_step': self.best_val_step,
        }
        with open(os.path.join(path, 'train_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.best_val_step = checkpoint.get('best_val_step', 0)
    
    def save_final_model(self, path: str):
        """保存最终模型"""
        self.model.save(path)
        
        # 保存训练配置
        training_info = {
            'training_steps': self.global_step,
            'epochs': self.epoch + 1,
            'learning_rate': self.training_config.learning_rate,
            'best_val_accuracy': round(self.best_val_accuracy, 4),
            'best_val_step': self.best_val_step,
        }
        with open(os.path.join(path, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"最终模型已保存: {path}")


# 注册到工厂
TrainableRouterFactory.register_trainer('hybrid_representation_fusion')(HybridRepresentationFusionTrainer)
TrainableRouterFactory.register_trainer('hybrid_rep_fusion')(HybridRepresentationFusionTrainer)
