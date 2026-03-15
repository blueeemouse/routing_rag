"""
FusionSoftLabelTrainer实现

【重要】此类支持多分类任务，配合融合模型（如 GatedFusionModel）使用

与 BinarySoftLabelTrainer 的区别：
================================================================================

| 特性             | BinarySoftLabelTrainer      | FusionSoftLabelTrainer         |
|------------------|----------------------------|--------------------------------|
| 分类类型         | 仅二分类                   | 支持多分类                     |
| 模型输入         | queries                    | input_ids, attention_mask, queries |
| 模型输出         | (batch, 1) 单值 logits     | (batch, num_strategies) logits |
| 损失函数         | BCEWithLogitsLoss          | CrossEntropyLoss (支持软标签)  |
| 软标签格式       | 单值 (float)               | 向量 (tensor)                  |

CrossEntropyLoss 支持软标签：
================================================================================

PyTorch >= 1.10 的 CrossEntropyLoss 直接支持概率分布作为 target：
```python
loss_fn = nn.CrossEntropyLoss()
logits = model(input_ids, attention_mask, queries)  # (batch, num_classes)
soft_labels = torch.tensor([[0.1, 0.7, 0.2], ...])  # (batch, num_classes)
loss = loss_fn(logits, soft_labels)  # 直接支持！
```

配合使用：
- FusionSoftLabelDataset（返回 input_ids, attention_mask, soft_label向量）
- GatedFusionModel（语义特征 + 统计特征融合）

================================================================================
"""

import os
import json
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..base_trainer import BaseTrainer
from ..factory import TrainableRouterFactory
from ..config import TrainableRouterConfig


class FusionSoftLabelTrainer(BaseTrainer):
    """
    融合模型软标签训练器
    
    【特点】支持多分类，配合融合模型使用
    
    特点：
    - 使用 CrossEntropyLoss 进行多分类（支持软标签）
    - 支持软标签向量 (soft_label ∈ [0, 1]^num_strategies, sum = 1)
    - 评估时使用 argmax 获取预测
    - 主要用于融合模型 (如 GatedFusionModel)
    
    数据要求：
    - batch['input_ids']: token IDs
    - batch['attention_mask']: 注意力掩码
    - batch['queries']: query 文本列表（用于统计特征提取）
    - batch['soft_label']: 软标签向量 (batch, num_strategies)
    - batch['label']: 硬标签 (用于评估)
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
            model: 模型 (支持融合式forward)
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
        
        # Debug 模式
        self.debug_mode = os.getenv("DEBUG_ROUTER", "false").lower() == "true"
        
        # 跟踪最佳性能
        self.best_val_accuracy = float('-inf')
        self.best_val_step = 0
        self.best_val_loss = float('inf')
        self.best_val_loss_step = 0
        
        print(f"FusionSoftLabelTrainer 初始化完成")
        print(f"  策略数量: {self.model.num_strategies}")
        print(f"  策略名称: {self.model.strategy_names}")
    
    def _init_optimizer(self):
        """初始化优化器"""
        lr = self.training_config.learning_rate
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
        计算软标签多分类损失
        
        Args:
            batch: 批次数据
                - input_ids: token IDs
                - attention_mask: 注意力掩码
                - queries: query文本列表
                - soft_label: 软标签向量 (batch, num_strategies)
                - label: 硬标签 (可选，用于评估)
                
        Returns:
            损失值
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        queries = batch['queries']
        
        # 前向传播 - 融合模型接收三个输入
        logits = self.model.forward(input_ids, attention_mask, queries)  # (batch, num_strategies)
        
        # 获取软标签向量（优先使用 soft_label，否则从 scores 或 label）
        if 'soft_label' in batch:
            soft_labels = batch['soft_label'].to(self.device).float()  # (batch, num_strategies)
        elif 'scores' in batch:
            soft_labels = batch['scores'].to(self.device).float()
        else:
            # 从硬标签生成 one-hot 编码作为软标签
            labels = batch['label'].to(self.device).long()
            soft_labels = F.one_hot(labels, num_classes=logits.size(-1)).float()
        
        # CrossEntropyLoss 支持软标签 (PyTorch >= 1.10)
        # 注意：CrossEntropyLoss 期望 target 是概率分布时不需要 squeeze
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, soft_labels)
        
        # 记录
        if self.global_step % self.training_config.eval_steps == 0:
            probs = F.softmax(logits, dim=-1)
            loss_str = f"Step {self.global_step}: Loss = {loss:.6f}, Probs range = [{probs.min():.4f}, {probs.max():.4f}]"
            if self.logger:
                self.logger.info(loss_str)
            
            if self.debug_mode:
                print(f"[DEBUG] soft_labels mean={soft_labels.mean():.4f}, std={soft_labels.std():.4f}")
                print(f"[DEBUG] logits mean={logits.mean():.4f}, std={logits.std():.4f}")
        
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
                    
                    if self.logger:
                        self.logger.info(f"评估 (Step {self.global_step}): Acc={val_acc:.4f}, Loss={val_loss:.4f}")
                        
                        routing_dist = val_metrics.get('routing_distribution', {})
                        self.logger.info("预测路由分布:")
                        for strategy, ratio in sorted(routing_dist.items()):
                            self.logger.info(f"    {strategy}: {ratio*100:.2f}%")
                        
                        strategy_acc = val_metrics.get('strategy_accuracy', {})
                        self.logger.info("各策略召回率:")
                        for strategy, acc in sorted(strategy_acc.items()):
                            self.logger.info(f"    {strategy}: {acc:.4f}")
                        
                        self.logger.info("")
                    
                    # 保存最佳模型 (基于 accuracy)
                    if val_acc > self.best_val_accuracy:
                        self.best_val_accuracy = val_acc
                        self.best_val_step = self.global_step
                        
                        checkpoint_path = f"{self.output_dir}/checkpoint_best_val"
                        self.save_checkpoint(checkpoint_path)
                        
                        if self.logger:
                            self.logger.info(f"🏆 新的最佳Val性能！Step={self.global_step}, Acc={val_acc:.4f}")
                            self.logger.info("")
        
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
        all_soft_labels = []
        total_loss = 0.0
        num_batches = 0
        
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ascii=True):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                queries = batch['queries']
                
                # 前向传播
                logits = self.model.forward(input_ids, attention_mask, queries)
                
                # 获取软标签（优先使用 soft_label，否则从 scores 或 label）
                if 'soft_label' in batch:
                    soft_labels = batch['soft_label'].to(self.device).float()
                elif 'scores' in batch:
                    # 从 scores 作为软标签（硬标签数据集）
                    soft_labels = batch['scores'].to(self.device).float()
                else:
                    # 从硬标签生成 one-hot 编码
                    labels = batch['label'].to(self.device).long()
                    soft_labels = F.one_hot(labels, num_classes=logits.size(-1)).float()
                
                # 获取硬标签（从 label 或 scores 推断）
                if 'label' in batch:
                    labels = batch['label'].to(self.device).long()
                else:
                    labels = soft_labels.argmax(dim=-1).long()
                
                # 计算损失
                loss = loss_fn(logits, soft_labels)
                total_loss += loss.item()
                num_batches += 1
                
                # 预测 (argmax)
                predictions = logits.argmax(dim=-1)
                
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_soft_labels.extend(soft_labels.cpu().tolist())
        
        # 计算准确率 (基于硬标签)
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0
        
        # 计算各策略召回率
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
        
        # 计算软标签相关指标
        all_soft_labels = np.array(all_soft_labels)
        all_predictions = np.array(all_predictions)
        
        # 计算预期校准误差 (ECE) - 衡量预测概率与实际准确率的差异
        ece = self._compute_ece(all_predictions, all_labels, all_soft_labels)
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'strategy_accuracy': strategy_accuracy,
            'routing_distribution': routing_distribution,
            'label_distribution': label_distribution,
            'num_samples': len(all_labels),
            'ece': ece,
        }
        
        # 打印评估结果
        print(f"\n{'='*80}")
        print(f"评估结果 (总样本: {len(all_labels)})")
        print(f"{'='*80}")
        print(f"  整体准确率: {accuracy:.4f}")
        print(f"  平均损失: {metrics['loss']:.4f}")
        print(f"  ECE (校准误差): {ece:.4f}")
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
    
    def _compute_ece(self, predictions, labels, soft_labels, n_bins=10):
        """
        计算预期校准误差 (Expected Calibration Error)
        
        Args:
            predictions: 预测标签
            labels: 真实标签
            soft_labels: 软标签（预测概率）
            n_bins: 分箱数量
            
        Returns:
            ECE 值
        """
        # 获取预测概率
        pred_probs = np.max(soft_labels, axis=1)
        correct = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (pred_probs >= bin_boundaries[i]) & (pred_probs < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(pred_probs[in_bin])
                avg_accuracy = np.mean(correct[in_bin])
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return ece
    
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
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_step': self.best_val_step,
        }
        with open(os.path.join(path, 'train_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        # 保存模型配置
        model_config = {
            'strategy_names': self.model.strategy_names,
            'num_strategies': self.model.num_strategies,
            'trainer_type': 'fusion_soft_label',
        }
        # 添加模型特定配置
        if hasattr(self.model, 'hidden_size'):
            model_config['hidden_size'] = self.model.hidden_size
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'backbone_name'):
            model_config['backbone_name'] = self.model.config.backbone_name
            
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
                'trainer_type': 'fusion_soft_label',
            }
            json.dump(training_info, f, indent=2)


# 注册到工厂
TrainableRouterFactory.register_trainer('fusion_soft_label')(FusionSoftLabelTrainer)
