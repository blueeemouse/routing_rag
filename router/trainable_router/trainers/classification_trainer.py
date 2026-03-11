"""
ClassificationTrainer实现

基于交叉熵损失的分类训练器
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

# TensorBoard 支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ClassificationTrainer(BaseTrainer):
    """分类训练器"""
    
    def __init__(self, model, config: TrainableRouterConfig, output_dir: str = "outputs", logger=None):
        """
        初始化

        Args:
            model: DCRouter模型
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
        
        # 监控当前batch data上的准确率的变量（用于debug）
        self.monitor_accuracy = os.getenv("MONITOR_ACC", "false").lower() == "true"
        # 这个用来监控router接收到的tensor，确认不是全0/全1这种诡异的值
        self.debug_mode = os.getenv("DEBUG_ROUTER", "false").lower() == "true"
        
        # 【新增】读取类别权重配置
        self.class_weights = getattr(self.training_config, 'class_weights', None)
        if self.class_weights:
            print(f"【Config】使用类别权重: {self.class_weights}")

        # 【新增】跟踪最佳性能
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
        
        # 后备保护：如果 total_steps <= 0，报错
        if total_steps <= 0:
            raise ValueError(
                f"training_steps={total_steps} 无效，调度器无法创建。 "
                f"请在训练配置中设置正确的 training_steps 或 epochs，"
                f"确保在创建 trainer 之前已经动态计算了总步数。"
            )
        
        # 原有的调度器初始化逻辑
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
        计算交叉熵分类损失（处理分数相同的情况）
        
        Args:
            batch: 批次数据
                - input_ids: token ids (可选)
                - attention_mask: 注意力掩码 (可选)
                - scores: 策略分数, shape: (batch_size, num_strategies)
                - queries: 查询文本列表 (用于 SentenceTransformer)
                
        Returns:
            损失值
        """
        scores = batch['scores'].to(self.device)
        
        # 前向传播获取query embedding - 统一使用transformers方式
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        query_emb = self.model.forward(input_ids, attention_mask)
        # print('#'*60)
        # print('using transformers forward method in training')
        # print('#'*60)

        # 新增，用于debug，看query_emb的值
        if self.debug_mode:
            mean_val = query_emb.mean().item()
            std_val = query_emb.std().item()
            max_val = query_emb.max().item()
            min_val = query_emb.min().item()
            print(f"\n[DEBUG Tensor] Step {self.global_step}: Mean={mean_val:.4f}, Std={std_val:.4f}, Max={max_val:.4f}, Min={min_val:.4f}")
            
            # 额外检查：看看 Batch 里前两个样本的相似度（用于判断特征是否混淆）
            # 如果相似度极高（>0.8），说明特征很难分
            cos_sim = torch.nn.functional.cosine_similarity(query_emb[0:1], query_emb[1:2]).item()
            print(f"[DEBUG Tensor] CosSim(Sample0, Sample1) = {cos_sim:.4f}")
        
        # 获取策略embedding
        strategy_emb = self.model.get_strategy_embeddings()
        
        # 计算相似度（作为logits）
        logits = self.model.compute_similarity(query_emb, strategy_emb)
        
        # 获取真实标签（处理分数相同的情况）
        # 假设: no_rag=0, naive_rag=1
        # no_rag_scores = scores[:, 0]  # no_rag分数
        # naive_rag_scores = scores[:, 1]  # naive_rag分数
        strategy_to_idx = {name: idx for idx, name in enumerate(self.model.strategy_names)}
        no_rag_idx = strategy_to_idx['no_rag']
        naive_rag_idx = strategy_to_idx['naive_rag']
        no_rag_scores = scores[:, no_rag_idx]
        naive_rag_scores = scores[:, naive_rag_idx]
        
        # 检测分数是否相同（考虑浮点数精度）
        score_diff = torch.abs(no_rag_scores - naive_rag_scores)
        is_tie = score_diff < 1e-6
        
        # 默认用argmax
        labels = scores.argmax(dim=-1)
        
        # 对分数相同的样本，强制设为no_rag (0)
        # labels = torch.where(is_tie, torch.zeros_like(labels), labels)
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
            
            # 从config获取权重（复用上面已经构建的strategy_to_idx）
            if self.class_weights:
                for strategy_name, weight in self.class_weights.items():
                    if strategy_name in strategy_to_idx:
                        idx = strategy_to_idx[strategy_name]
                        pos_weight[idx] = weight
                
                if self.debug_mode:
                    print(f"Using Weighted Loss: {self.class_weights}")
            
            loss_fn = torch.nn.CrossEntropyLoss(weight=pos_weight)
            loss = loss_fn(logits, labels)




            # 按eval_steps间隔记录训练loss（同时输出到控制台和文件）
        if self.global_step % self.training_config.eval_steps == 0:
            loss_str = f"Step {self.global_step}: Loss = {loss:.6f}, Logits range = [{logits.min():.4f}, {logits.max():.4f}]"
            if self.logger:
                self.logger.info(loss_str)
        
        # 【关键点2】不改返回值！但把logits存到self里
        # 使用detach()是为了切断计算图，省显存，且不影响loss反向传播
        # 只有当开关开启时才存，省去不必要的开销
        if self.monitor_accuracy:
            self._last_logits = logits.detach()
            self._last_labels = labels.detach()
        
        # # 【新增 2】打印当前Batch的原始 Loss
        # if self.debug_mode:
        #     print(f"[DEBUG Loss] Step {self.global_step}: Total_Loss={loss.item():.4f}")
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        训练一个epoch（计算梯度、更新参数、定期评估、定期保存）
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 使用基类的数据生成器，支持overfit模式
        batch_generator = self._get_training_batches(dataloader)
        
        pbar = tqdm(batch_generator, desc=f"Epoch {self.epoch + 1}", ascii=True)
        
        for batch in pbar:
            # 如果指定了 max_steps 并且已达到，则提前结束本 epoch
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
                # 使用验证数据加载器进行评估，如果没有则跳过
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
                            self.logger.info("")  # 空行分隔

                # TensorBoard 记录评估指标
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('Accuracy/val', val_metrics.get('accuracy', 0), self.global_step)
                    self.tensorboard_writer.add_scalar('Loss/val', val_metrics.get('loss', 0), self.global_step)

                    # 记录每个策略的召回率
                    strategy_acc = val_metrics.get('strategy_accuracy', {})
                    for strategy, acc in strategy_acc.items():
                        self.tensorboard_writer.add_scalar(f'Recall/val_{strategy}', acc, self.global_step)

                    # 记录路由分布
                    routing_dist = val_metrics.get('routing_distribution', {})
                    for strategy, ratio in routing_dist.items():
                        self.tensorboard_writer.add_scalar(f'RoutingDistribution/val_{strategy}', ratio, self.global_step)
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0
        }



    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型

        Args:
            dataloader: 数据加载器

        Returns:
            评估指标字典，包含以下key：
            - 'accuracy': 整体准确率
            - 'loss': 平均损失值
            - 'strategy_accuracy': 各策略的召回率（针对真实标签为该策略的样本）
            - 'routing_distribution': 预测路由分布（模型预测的各策略比例）
            - 'label_distribution': 真实标签分布（验证集中各策略的真实比例）
            - 'num_samples': 验证样本总数
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

                # 前向传播 - 统一使用transformers方式
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                query_emb = self.model.forward(input_ids, attention_mask)

                # 获取策略embedding
                strategy_emb = self.model.get_strategy_embeddings()

                # 计算相似度
                similarity = self.model.compute_similarity(query_emb, strategy_emb)

                # 预测
                predictions = similarity.argmax(dim=-1)
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

        # 计算每个策略的准确率（召回率）
        strategy_accuracy = {}
        for i, strategy in enumerate(self.model.strategy_names):
            mask = [l == i for l in all_labels]
            if any(mask):
                strategy_correct = sum(p == l for p, l, m in zip(all_predictions, all_labels, mask) if m)
                strategy_total = sum(mask)
                strategy_accuracy[strategy] = strategy_correct / strategy_total if strategy_total > 0 else 0.0

        # 计算路由分布（预测结果中各策略的比例）
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

        print("total_loss:", total_loss)
        print("total_num_batches:", num_batches)
        # 打印详细评估信息
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
        """
        删除checkpoint

        Args:
            path: checkpoint路径
        """
        import shutil
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                # 不记录删除日志（删除次数太多，日志冗余）
        except Exception as e:
            # 只记录失败
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
TrainableRouterFactory.register_trainer('classification')(ClassificationTrainer)
