"""
BinarySoftLabelTrainer实现

【重要】此类仅支持二分类任务（no_rag vs naive_rag）

用于二分类软标签训练的训练器。

与 ClassificationTrainer / StatisticalTrainer 的区别：
================================================================================

1. **分类器式模型** (StatisticalRouterModel等)
   - forward(query) 直接返回 logits (batch_size, 1)
   - 内部提取手工特征，通过MLP输出
   
2. **Embedding相似度式模型** (DCRouterModel等)  
   - forward(input_ids, attention_mask) 返回 query_embedding
   - 需要 compute_similarity(query_emb, strategy_emb) 计算相似度作为 logits
   - logits 形状 (batch_size, num_strategies)

本 Trainer 主要针对**分类器式模型**设计：
- 使用 BCEWithLogitsLoss 进行二分类训练
- 支持软标签 (soft_label ∈ [0, 1]) 而非硬标签
- 评估时使用 sigmoid + threshold 判断

软标签含义：
- soft_label → 0: no_rag 更好
- soft_label → 0.5: 两者差不多 (tie)
- soft_label → 1: naive_rag 更好

与 FusionSoftLabelTrainer 的区别：
- 本类：仅支持二分类，输出单值logit，使用 BCEWithLogitsLoss
- FusionSoftLabelTrainer：支持多分类，输出多值logits，使用 CrossEntropyLoss

================================================================================
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


class BinarySoftLabelTrainer(BaseTrainer):
    """
    二分类软标签训练器
    
    【限制】仅支持二分类任务（no_rag vs naive_rag）
    
    特点：
    - 使用 BCEWithLogitsLoss 进行二分类
    - 支持软标签 (soft_label ∈ [0, 1])
    - 评估时使用 sigmoid + threshold
    - 主要用于分类器式模型 (如 StatisticalRouterModel)
    
    数据要求：
    - batch['queries']: query 文本列表
    - batch['soft_label']: 软标签 (0~1)
    - batch['scores']: 兼容字段，用于获取硬标签 (评估时)
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
            model: 模型 (支持分类器式forward)
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
        
        # 软标签阈值 (评估时使用)
        self.threshold = getattr(self.training_config, 'soft_label_threshold', 0.5)
        
        # Debug 模式
        self.debug_mode = os.getenv("DEBUG_ROUTER", "false").lower() == "true"
        
        # 跟踪最佳性能
        self.best_val_accuracy = float('-inf')
        self.best_val_step = 0
        self.best_val_loss = float('inf')
        self.best_val_loss_step = 0
        
        print(f"SoftLabelTrainer 初始化完成")
        print(f"  软标签阈值: {self.threshold}")
    
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
        计算软标签二分类损失
        
        Args:
            batch: 批次数据
                - queries: query文本列表 (分类器式模型)
                - soft_label: 软标签 (0~1), shape: (batch_size,)
                - scores: 策略分数 (兼容字段, 用于获取硬标签)
                
        Returns:
            损失值
        """
        queries = batch.get('queries', [])
        
        if not queries:
            raise ValueError("Batch中缺少'queries'字段")
        
        # 前向传播 - 分类器式模型直接接收 queries
        logits = self.model.forward(queries)  # (batch_size, 1)
        
        # 获取软标签
        if 'soft_label' in batch:
            soft_labels = batch['soft_label'].to(self.device).float()  # (batch_size,)
        else:
            # 兼容：从 scores 推断软标签
            scores = batch['scores'].to(self.device)
            # 假设 scores[1] 是 naive_rag 的分数
            soft_labels = scores[:, 1].float()
        
        # BCEWithLogitsLoss (适用于软标签)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.squeeze(-1), soft_labels)
        
        # 记录
        if self.global_step % self.training_config.eval_steps == 0:
            loss_str = f"Step {self.global_step}: Loss = {loss:.6f}, Logits = [{logits.min():.4f}, {logits.max():.4f}]"
            if self.logger:
                self.logger.info(loss_str)
            
            if self.debug_mode:
                print(f"[DEBUG] soft_labels mean={soft_labels.mean():.4f}, std={soft_labels.std():.4f}")
        
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
        all_soft_labels = []  # 用于计算软标签相关指标
        total_loss = 0.0
        num_batches = 0
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ascii=True):
                queries = batch.get('queries', [])
                
                # 前向传播
                logits = self.model.forward(queries)
                
                # 获取软标签
                if 'soft_label' in batch:
                    soft_labels = batch['soft_label'].to(self.device).float()
                else:
                    scores = batch['scores'].to(self.device)
                    soft_labels = scores[:, 1].float()
                
                # 获取硬标签 (用于评估准确率)
                scores = batch['scores'].to(self.device)
                hard_labels = scores.argmax(dim=-1).long()  # 0 或 1
                
                # 计算损失
                loss = loss_fn(logits.squeeze(-1), soft_labels)
                total_loss += loss.item()
                num_batches += 1
                
                # 预测 (sigmoid + threshold)
                probs = torch.sigmoid(logits.squeeze(-1))
                predictions = (probs >= self.threshold).long()
                
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(hard_labels.cpu().tolist())
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
        
        # 按 soft_label 分组统计预测准确率
        soft_label_bins = {
            '< 0.3 (倾向no_rag)': (all_soft_labels < 0.3, 'no_rag should be predicted'),
            '0.3~0.7 (模糊)': ((all_soft_labels >= 0.3) & (all_soft_labels <= 0.7), 'uncertain'),
            '> 0.7 (倾向naive_rag)': (all_soft_labels > 0.7, 'naive_rag should be predicted'),
        }
        
        bin_accuracy = {}
        for bin_name, (mask, _) in soft_label_bins.items():
            if mask.sum() > 0:
                bin_correct = (all_predictions[mask] == (all_soft_labels[mask] > 0.5).astype(int)).sum()
                bin_accuracy[bin_name] = bin_correct / mask.sum()
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'strategy_accuracy': strategy_accuracy,
            'routing_distribution': routing_distribution,
            'label_distribution': label_distribution,
            'num_samples': len(all_labels),
            'bin_accuracy': bin_accuracy,
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
        print(f"\n  软标签分组预测准确率:")
        for bin_name, acc in bin_accuracy.items():
            print(f"    {bin_name}: {acc:.4f}")
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
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_step': self.best_val_step,
        }
        with open(os.path.join(path, 'train_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        # 保存模型配置
        model_config = {
            'strategy_names': self.model.strategy_names,
            'num_strategies': self.model.num_strategies,
            'threshold': self.threshold,
        }
        # 添加模型特定配置
        if hasattr(self.model, 'mlp_hidden_dims'):
            model_config['mlp_hidden_dims'] = self.model.mlp_hidden_dims
        if hasattr(self.model, 'threshold'):
            model_config['model_threshold'] = self.model.threshold
            
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
                'threshold': self.threshold,
            }
            json.dump(training_info, f, indent=2)


# 注册到工厂
TrainableRouterFactory.register_trainer('binary_soft_label')(BinarySoftLabelTrainer)

# 兼容旧名称（deprecated，将在未来版本移除）
SoftLabelTrainer = BinarySoftLabelTrainer
