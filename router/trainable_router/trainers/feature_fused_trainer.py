"""
FeatureFusedTrainer实现

基于特征融合的分类训练器
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

# TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class FeatureFusedTrainer(BaseTrainer):
    """
    特征融合训练器
    
    与ClassificationTrainer的关键区别：
    - 模型forward需要queries参数来提取手工特征
    - 使用分类器而非相似度计算
    """
    
    def __init__(self, model, config: TrainableRouterConfig, output_dir: str = "outputs", logger=None):
        """
        初始化

        Args:
            model: FeatureFusedRouterModel
            config: 配置
            output_dir: 输出目录
            logger: (可选) 标准 logging.Logger 对象
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

        # TensorBoard 记录器
        self.tensorboard_writer = None
        if TENSORBOARD_AVAILABLE:
            tensorboard_dir = os.path.join(output_dir, "tensorboard")
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard 日志将保存到: {tensorboard_dir}")
        else:
            print("TensorBoard 未安装，跳过 TensorBoard 记录")
        
        # Debug配置
        self.monitor_accuracy = os.getenv("MONITOR_ACC", "false").lower() == "true"
        self.debug_mode = os.getenv("DEBUG_ROUTER", "false").lower() == "true"
        
        # 类别权重
        self.class_weights = getattr(self.training_config, 'class_weights', None)
        if self.class_weights:
            print(f"【Config】使用类别权重: {self.class_weights}")

        # 跟踪最佳性能
        self.best_train_loss = float('inf')
        self.best_train_step = 0
        self.best_val_accuracy = float('-inf')  # 初始化为负无穷，确保第一次评估能正确记录
        self.best_val_step = 0
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """初始化优化器"""
        lr = self.training_config.learning_rate
        
        if self.training_config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                **self.training_config.optimizer_kwargs
            )
        elif self.training_config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                **self.training_config.optimizer_kwargs
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                **self.training_config.optimizer_kwargs
            )
        
        return optimizer
    
    def _init_scheduler(self):
        """初始化学习率调度器"""
        total_steps = self.training_config.training_steps
        
        if total_steps <= 0:
            raise ValueError(
                f"training_steps={total_steps} 无效。"
                f"请在训练配置中设置正确的 training_steps。"
            )
        
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
        计算交叉熵分类损失
        
        Args:
            batch: 批次数据
                - input_ids: token ids
                - attention_mask: 注意力掩码
                - scores: 策略分数, shape: (batch_size, num_strategies)
                - queries: 查询文本列表
                
        Returns:
            损失值
        """
        scores = batch['scores'].to(self.device)
        queries = batch['queries']  # 原始query列表
        
        # 前向传播（需要queries参数）
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        logits = self.model(input_ids, attention_mask, queries)  # ⭐ 关键区别：传递queries

        # Debug信息
        if self.debug_mode:
            mean_val = logits.mean().item()
            std_val = logits.std().item()
            print(f"\n[DEBUG] Step {self.global_step}: Logits Mean={mean_val:.4f}, Std={std_val:.4f}")
        
        # 获取真实标签
        strategy_to_idx = {name: idx for idx, name in enumerate(self.model.strategy_names)}
        
        # 处理分数相同的情况
        labels = scores.argmax(dim=-1)
        
        # 检测分数是否相同
        if len(self.model.strategy_names) == 2:
            no_rag_idx = strategy_to_idx.get('no_rag', 0)
            naive_rag_idx = strategy_to_idx.get('naive_rag', 1)
            score_diff = torch.abs(scores[:, no_rag_idx] - scores[:, naive_rag_idx])
            is_tie = score_diff < 1e-6
            labels = torch.where(is_tie, torch.tensor(no_rag_idx, device=self.device), labels)
        
        # 计算交叉熵损失
        # 检查是否使用样本权重
        if 'sample_weights' in batch:
            # 使用样本权重
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(logits, labels)  # shape: (batch_size,)
            sample_weights = batch['sample_weights'].to(self.device)
            loss = (losses * sample_weights).sum() / sample_weights.sum()  # 加权平均
            
            if self.debug_mode:
                print(f"Using Sample Weights: mean={sample_weights.mean():.4f}, min={sample_weights.min():.4f}, max={sample_weights.max():.4f}")
        else:
            # 使用类别权重（或默认无权重）
            num_strategies = len(self.model.strategy_names)
            pos_weight = torch.ones(num_strategies).to(self.device)
            
            # 从config获取权重
            if self.class_weights:
                for strategy_name, weight in self.class_weights.items():
                    if strategy_name in strategy_to_idx:
                        idx = strategy_to_idx[strategy_name]
                        pos_weight[idx] = weight
                
                if self.debug_mode:
                    print(f"Using Weighted Loss: {self.class_weights}")
            
            loss_fn = torch.nn.CrossEntropyLoss(weight=pos_weight)
            loss = loss_fn(logits, labels)

        # 记录训练loss
        if self.global_step % self.training_config.eval_steps == 0:
            loss_str = f"Step {self.global_step}: Loss = {loss:.6f}"
            if self.logger:
                self.logger.info(loss_str)
        
        # Monitor accuracy
        if self.monitor_accuracy:
            self._last_logits = logits.detach()
            self._last_labels = labels.detach()
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            max_steps: 最大训练步数
            
        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 使用基类的数据生成器
        batch_generator = self._get_training_batches(dataloader)
        
        pbar = tqdm(batch_generator, desc=f"Epoch {self.epoch + 1}", ascii=True)
        
        for batch in pbar:
            # 如果达到max_steps，提前结束
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

            # TensorBoard 记录
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('Loss/train', loss.item(), self.global_step)
                if self.scheduler is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tensorboard_writer.add_scalar('LearningRate', current_lr, self.global_step)

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

                    # 判断并更新最佳val性能，如果是最佳则保存checkpoint
                    if val_acc > self.best_val_accuracy:
                        self.best_val_accuracy = val_acc
                        self.best_val_step = self.global_step
                        
                        # 保存最佳val checkpoint
                        checkpoint_path = f"{self.output_dir}/checkpoint_best_val"
                        self.save_checkpoint(checkpoint_path)
                        
                        if self.logger:
                            self.logger.info(f"🏆 新的最佳Val性能！Step={self.global_step}, Acc={val_acc:.4f}")
                            self.logger.info(f"最佳Val模型已保存到: {checkpoint_path}")

                # TensorBoard 记录评估指标
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('Accuracy/val', val_metrics.get('accuracy', 0), self.global_step)
                    self.tensorboard_writer.add_scalar('Loss/val', val_metrics.get('loss', 0), self.global_step)

                    # 记录每个策略的召回率
                    strategy_acc = val_metrics.get('strategy_accuracy', {})
                    for strategy, acc in strategy_acc.items():
                        self.tensorboard_writer.add_scalar(f'Recall/val_{strategy}', acc, self.global_step)
        
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

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ascii=True):
                scores = batch['scores'].to(self.device)
                queries = batch.get('queries', [])

                # 前向传播
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self.model(input_ids, attention_mask, queries)  # ⭐ 传递queries

                # 预测
                predictions = logits.argmax(dim=-1)
                labels = scores.argmax(dim=-1)

                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                # 计算损失
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        # 计算准确率
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0

        # 计算每个策略的准确率
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

        # 计算真实标签分布
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

        # 打印评估信息
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
        
        # 保存模型配置（用于推理加载）
        model_config = {
            'strategy_names': self.model.strategy_names,
            'hidden_size': self.model.hidden_size,
            'num_strategies': self.model.num_strategies,
        }
        # 添加可选配置（如果模型有这些属性）
        if hasattr(self.model, 'backbone_name'):
            model_config['backbone_name'] = self.model.backbone_name if hasattr(self.model.backbone, 'name_or_path') else self.config.model.backbone_name
        if hasattr(self.model, 'temperature'):
            model_config['temperature'] = self.model.temperature
        if hasattr(self.model, 'similarity_function'):
            model_config['similarity_function'] = self.model.similarity_function
            
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)

    def _delete_checkpoint(self, path: str):
        """删除checkpoint"""
        import shutil
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"删除checkpoint失败: {path}, 错误: {e}")

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
    
    def close(self):
        """关闭资源"""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            print("TensorBoard writer 已关闭")


# 注册到工厂
TrainableRouterFactory.register_trainer('feature_fused')(FeatureFusedTrainer)
