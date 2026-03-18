"""
DecisionRouterTrainer实现

用于训练决策式路由器模型的训练器。

训练目标：
1. Q回归损失：MSE(Q_pred, Q_true)
2. Cost回归损失：MSE(cost_pred, cost_true)
3. 可选：决策损失（cross-entropy或ranking）

损失函数：
total_loss = q_loss_weight * Q_loss + cost_loss_weight * Cost_loss + decision_loss_weight * Decision_loss

评估指标：
1. 路由准确率（基于utility决策的准确率）
2. Q预测误差（MSE）
3. Cost预测误差（MSE）
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


class DecisionRouterTrainer(BaseTrainer):
    """
    决策式路由器训练器
    
    特点：
    1. 多任务学习：同时预测Q和cost
    2. 回归损失：MSE
    3. 评估路由决策准确率
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
            model: 模型 (DecisionRouterModel)
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
        
        # 损失权重（从配置或kwargs获取）
        self.q_loss_weight = getattr(self.training_config, 'q_loss_weight', 1.0)
        self.cost_loss_weight = getattr(self.training_config, 'cost_loss_weight', 1.0)
        self.decision_loss_weight = getattr(self.training_config, 'decision_loss_weight', 0.0)
        
        # Debug模式
        self.debug_mode = os.getenv("DEBUG_ROUTER", "false").lower() == "true"
        
        # 跟踪最佳性能
        self.best_val_accuracy = float('-inf')
        self.best_val_step = 0
        self.best_val_loss = float('inf')
        self.best_val_loss_step = 0
        
        # 获取需要检索的策略（用于cost损失计算）
        self.need_retrieval_mask = self._get_retrieval_mask()
        
        print(f"DecisionRouterTrainer 初始化完成")
        print(f"  Q loss weight: {self.q_loss_weight}")
        print(f"  Cost loss weight: {self.cost_loss_weight}")
        print(f"  Decision loss weight: {self.decision_loss_weight}")
        print(f"  Retrieval mask: {self.need_retrieval_mask}")
    
    def _get_retrieval_mask(self) -> torch.Tensor:
        """
        获取需要检索的策略mask
        
        Returns:
            mask tensor, shape: (num_strategies,)
            - 1表示需要检索（有cost）
            - 0表示不需要检索（cost=0）
        """
        mask = torch.ones(self.model.num_strategies)
        for i, strategy in enumerate(self.model.strategy_names):
            if strategy == 'no_rag':
                mask[i] = 0.0
        return mask.to(self.device)
    
    def _init_optimizer(self):
        """初始化优化器"""
        lr = self.training_config.learning_rate
        weight_decay = self.training_config.optimizer_kwargs.get('weight_decay', 0.01) if self.training_config.optimizer_kwargs else 0.01
        
        # 区分backbone和其他参数的学习率
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': lr * 0.1},  # backbone使用较小学习率
            {'params': other_params, 'lr': lr},
        ]
        
        if self.training_config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        
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
    
    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            batch: 批次数据
                - input_ids: token ids
                - attention_mask: 注意力掩码
                - Q: 各策略的真实Q值 (B, num_strategies)
                - costs: 各策略的真实cost值 (B, num_strategies)
                - label: 最优策略索引
                
        Returns:
            损失字典 {'total': total_loss, 'q_loss': ..., 'cost_loss': ...}
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 前向传播
        outputs = self.model.forward(input_ids, attention_mask)
        Q_pred = outputs['Q_pred']  # (B, num_strategies)
        cost_pred = outputs['cost_pred']  # (B, num_strategies)
        utility_pred = outputs['utility']  # (B, num_strategies)
        
        # 获取目标值（已经是tensor）
        Q_true = batch['Q'].to(self.device)
        cost_true = batch['costs'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # 1. Q回归损失
        q_loss = nn.functional.mse_loss(Q_pred, Q_true)
        
        # 2. Cost回归损失（仅对需要检索的策略计算）
        # no_rag的cost始终为0，所以只对其他策略计算损失
        cost_mask = self.need_retrieval_mask.unsqueeze(0)  # (1, num_strategies)
        masked_cost_pred = cost_pred * cost_mask
        masked_cost_true = cost_true * cost_mask
        
        # 只计算非零部分的损失
        if cost_mask.sum() > 0:
            cost_loss = nn.functional.mse_loss(masked_cost_pred, masked_cost_true)
        else:
            cost_loss = torch.tensor(0.0, device=self.device)
        
        # 3. 决策损失（可选）
        decision_loss = torch.tensor(0.0, device=self.device)
        if self.decision_loss_weight > 0:
            # 使用cross-entropy损失，基于utility预测
            decision_loss = nn.functional.cross_entropy(utility_pred, labels)
        
        # 总损失
        total_loss = (
            self.q_loss_weight * q_loss +
            self.cost_loss_weight * cost_loss +
            self.decision_loss_weight * decision_loss
        )
        
        # 记录（定期）
        if self.global_step % self.training_config.eval_steps == 0:
            if self.logger:
                self.logger.info(
                    f"Step {self.global_step}: "
                    f"total={total_loss:.6f}, "
                    f"q_loss={q_loss:.6f}, "
                    f"cost_loss={cost_loss:.6f}, "
                    f"decision_loss={decision_loss:.6f}"
                )
        
        return {
            'total': total_loss,
            'q_loss': q_loss,
            'cost_loss': cost_loss,
            'decision_loss': decision_loss,
        }
    
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
        total_q_loss = 0.0
        total_cost_loss = 0.0
        num_batches = 0
        
        batch_generator = self._get_training_batches(dataloader)
        pbar = tqdm(batch_generator, desc=f"Epoch {self.epoch + 1}", ascii=True)
        
        for batch in pbar:
            if max_steps is not None and self.global_step >= max_steps:
                break
            
            # 计算损失
            loss_dict = self.compute_loss(batch)
            loss = loss_dict['total']
            
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
            total_q_loss += loss_dict['q_loss'].item()
            total_cost_loss += loss_dict['cost_loss'].item()
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
                        self.logger.info(f"  Q_MSE={val_metrics.get('q_mse', 0):.6f}, Cost_MSE={val_metrics.get('cost_mse', 0):.6f}")
                        
                        routing_dist = val_metrics.get('routing_distribution', {})
                        self.logger.info("  预测路由分布:")
                        for strategy, ratio in sorted(routing_dist.items()):
                            self.logger.info(f"    {strategy}: {ratio*100:.2f}%")
                        
                        self.logger.info("")
                    
                    # 保存最佳模型
                    if val_acc > self.best_val_accuracy:
                        self.best_val_accuracy = val_acc
                        self.best_val_step = self.global_step
                        
                        checkpoint_path = f"{self.output_dir}/checkpoint_best_val"
                        self.save_checkpoint(checkpoint_path)
                        
                        if self.logger:
                            self.logger.info(f"🏆 新的最佳Val性能！Step={self.global_step}, Acc={val_acc:.4f}")
                            self.logger.info("")
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'q_loss': total_q_loss / num_batches if num_batches > 0 else 0.0,
            'cost_loss': total_cost_loss / num_batches if num_batches > 0 else 0.0,
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
        all_Q_pred = []
        all_Q_true = []
        all_cost_pred = []
        all_cost_true = []
        all_utility_pred = []
        
        total_loss = 0.0
        total_q_loss = 0.0
        total_cost_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ascii=True):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = self.model.forward(input_ids, attention_mask)
                
                Q_pred = outputs['Q_pred']
                cost_pred = outputs['cost_pred']
                utility_pred = outputs['utility']
                
                # 获取目标值（已经是tensor）
                Q_true = batch['Q'].to(self.device)
                cost_true = batch['costs'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 计算损失
                loss_dict = self.compute_loss(batch)
                total_loss += loss_dict['total'].item()
                total_q_loss += loss_dict['q_loss'].item()
                total_cost_loss += loss_dict['cost_loss'].item()
                num_batches += 1
                
                # 收集预测结果
                predicted_indices = utility_pred.argmax(dim=-1)
                
                all_predictions.extend(predicted_indices.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_Q_pred.extend(Q_pred.cpu().tolist())
                all_Q_true.extend(Q_true.cpu().tolist())
                all_cost_pred.extend(cost_pred.cpu().tolist())
                all_cost_true.extend(cost_true.cpu().tolist())
                all_utility_pred.extend(utility_pred.cpu().tolist())
        
        # 计算准确率
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
        
        # 计算MSE
        all_Q_pred = np.array(all_Q_pred)
        all_Q_true = np.array(all_Q_true)
        all_cost_pred = np.array(all_cost_pred)
        all_cost_true = np.array(all_cost_true)
        
        q_mse = np.mean((all_Q_pred - all_Q_true) ** 2)
        cost_mse = np.mean((all_cost_pred - all_cost_true) ** 2)
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'q_loss': total_q_loss / num_batches if num_batches > 0 else 0.0,
            'cost_loss': total_cost_loss / num_batches if num_batches > 0 else 0.0,
            'q_mse': q_mse,
            'cost_mse': cost_mse,
            'strategy_accuracy': strategy_accuracy,
            'routing_distribution': routing_distribution,
            'label_distribution': label_distribution,
            'num_samples': len(all_labels),
        }
        
        # 打印评估结果
        print(f"\n{'='*80}")
        print(f"评估结果 (总样本: {len(all_labels)})")
        print(f"{'='*80}")
        print(f"  路由准确率: {accuracy:.4f}")
        print(f"  平均损失: {metrics['loss']:.4f}")
        print(f"  Q MSE: {q_mse:.6f}")
        print(f"  Cost MSE: {cost_mse:.6f}")
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
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_step': self.best_val_step,
            'q_loss_weight': self.q_loss_weight,
            'cost_loss_weight': self.cost_loss_weight,
            'decision_loss_weight': self.decision_loss_weight,
        }
        with open(os.path.join(path, 'train_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
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
                'q_loss_weight': self.q_loss_weight,
                'cost_loss_weight': self.cost_loss_weight,
                'decision_loss_weight': self.decision_loss_weight,
            }
            json.dump(training_info, f, indent=2)


# 注册到工厂
TrainableRouterFactory.register_trainer('decision_router')(DecisionRouterTrainer)
