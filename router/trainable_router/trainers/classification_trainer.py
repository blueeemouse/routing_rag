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
    
    def __init__(self, model, config: TrainableRouterConfig, output_dir: str = "outputs"):
        """
        初始化

        Args:
            model: DCRouter模型
            config: 配置
            output_dir: 输出目录
        """
        super().__init__(model, config, output_dir)

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

        # 【新增】读取是否使用加权Loss的开关
        self.use_weighted_loss = os.getenv("USE_WEIGHTED_LOSS", "false").lower() == "true"
        
        # 如果开启加权，打印提示
        if self.use_weighted_loss:
            print("【Config】已启用加权交叉熵损失 (USE_WEIGHTED_LOSS=true)")
    
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
        # loss_fn = torch.nn.CrossEntropyLoss() # 原来仅用普通交叉熵损失
        # 【修改点】根据开关选择Loss函数
        if self.use_weighted_loss:
            # 创建权重张量
            num_strategies = len(self.model.strategy_names) # strategy_names就是个列表，里面是所有候选策略的名称
            pos_weight = torch.ones(num_strategies).to(self.device)
            
            # 核心修正：给 no_rag (索引0) 更高的权重
            # 因为你之前的 log 显示模型总是漏掉 no_rag（召回率 0.36），说明它倾向于把 no_rag 判成 naive_rag
            # 加大 Class 0 的权重，可以惩罚这种"误判"
            pos_weight[no_rag_idx] = 3.0  # 可以尝试 2.0, 3.0, 5.0
            pos_weight[naive_rag_idx] = 1.0
            
            loss_fn = torch.nn.CrossEntropyLoss(weight=pos_weight)
            
            if self.debug_mode:
                # self.debug_logger.info(f"Using Weighted Loss: no_rag_weight={pos_weight[no_rag_idx].item()}, naive_rag_weight={pos_weight[naive_rag_idx].item()}")
                print(f"Using Weighted Loss: no_rag_weight={pos_weight[no_rag_idx].item()}, naive_rag_weight={pos_weight[naive_rag_idx].item()}")
        else:
            # 标准的交叉熵
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        # 添加这行打印
        if self.global_step % 10 == 0:  # 每10步打印一次
            print(f"Step {self.global_step}: Loss = {loss:.6f}, Logits range = [{logits.min():.4f}, {logits.max():.4f}]")
        
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
        训练一个epoch
        
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
                    print(f"Step {self.global_step}: Val Acc = {val_metrics.get('accuracy', 0):.4f}")

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

            # 定期保存检查点
            if self.global_step % self.training_config.save_steps == 0:
                checkpoint_path = f"{self.output_dir}/checkpoint_step_{self.global_step}"
                self.save_checkpoint(checkpoint_path)
                print(f"Step {self.global_step}: Checkpoint saved to {checkpoint_path}")
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
    
    # def train_epoch(self, dataloader: DataLoader, max_steps: Optional[int] = None) -> Dict[str, float]:
    #     """
    #     训练一个epoch（只用一个batch来过拟合测试）
        
    #     Args:
    #         dataloader: 数据加载器
            
    #     Returns:
    #         训练指标（一个字典）
    #     """
    #     self.model.train()
    #     total_loss = 0.0
    #     num_batches = 0
        
    #     # 只取第一个batch用于过拟合测试
    #     first_batch = None
    #     for batch in dataloader:
    #         first_batch = batch
    #         break
        
    #     if first_batch is None:
    #         return {'loss': 0.0}
        
    #     # 打印batch信息
    #     print(f"\n{'='*80}")
    #     print(f"使用过拟合模式：只用一个batch，大小 = {len(first_batch['queries'])}")
    #     print(f"Batch questions: {first_batch['queries'][:3]}...")  # 打印前3个问题
    #     print("Batch scores shape:", first_batch['scores'].shape)
    #     # print(f"Batch scores shape: {torch.stack(first_batch['scores']).shape}")
    #     print(f"{'='*80}\n")
        
    #     # 训练多个steps（相当于多个epoch但用同一个batch）
    #     num_overfit_steps = 100  # 训练100步来尝试过拟合
    #     pbar = tqdm(range(num_overfit_steps), desc=f"Epoch {self.epoch + 1} (Overfit)", ascii=True)
        
    #     for step in pbar:
    #         batch = first_batch  # 始终使用第一个batch
    #         # 如果指定了 max_steps 并且已达到，则提前结束本 epoch
    #         # if max_steps is not None and self.global_step >= max_steps:
    #         #     break
                
    #         # 计算损失
    #         loss = self.compute_loss(batch)
            
    #         # 反向传播
    #         self.optimizer.zero_grad()
    #         loss.backward()
            
    #         # 梯度裁剪
    #         if self.training_config.max_grad_norm > 0:
    #             torch.nn.utils.clip_grad_norm_(
    #                 self.model.parameters(), 
    #                 self.training_config.max_grad_norm
    #             )
            
    #         self.optimizer.step()

    #         # 【关键点3】从这里"偷"出 logits 来算准确率
    #         if self.monitor_accuracy:
    #             # 验证一下是否存在（防御性编程）
    #             if hasattr(self, '_last_logits'):
    #                 preds = torch.argmax(self._last_logits, dim=-1)
    #                 correct = (preds == self._last_labels).sum().item()
    #                 acc = correct / len(self._last_labels)
                    
    #                 # 打印或者用 logger 记录
    #                 print(f"Step {step}: Acc/Train={acc:.4f}")

    #         if self.scheduler is not None:
    #             self.scheduler.step()

    #         self.global_step += 1
    #         total_loss += loss.item()
    #         num_batches += 1

    #         # TensorBoard 记录
    #         if self.tensorboard_writer is not None:
    #             self.tensorboard_writer.add_scalar('Loss/train', loss.item(), self.global_step)
    #             if self.scheduler is not None:
    #                 current_lr = self.optimizer.param_groups[0]['lr']
    #                 self.tensorboard_writer.add_scalar('LearningRate', current_lr, self.global_step)

    #         # 更新进度条
    #         pbar.set_postfix({
    #             'loss': f'{loss.item():.4f}',
    #             'step': self.global_step
    #         })

    #         # if step % 10 == 0:
    #         #     print(f"\nStep {step}:")
    #         #     print(f"  Loss: {loss.item():.6f}")
    #         #     print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    #         #     predictions = logits.argmax(dim=-1)
    #         #     labels = batch['scores'].to(self.device).argmax(dim=-1)
    #         #     accuracy = (predictions == labels).float().mean()
    #         #     print(f"  Accuracy: {accuracy:.4f}")
    #         #     print(f"  Predictions: {predictions[:8].tolist()}")
    #         #     print(f"  Labels: {labels[:8].tolist()}")
    #         #     print(f"  Same? {torch.equal(predictions, labels)}")

    #         # 定期评估
    #         if self.global_step % self.training_config.eval_steps == 0:
    #             # 使用验证数据加载器进行评估，如果没有则跳过
    #             if self.val_dataloader is not None:
    #                 val_metrics = self.evaluate(self.val_dataloader)
    #                 print(f"Step {self.global_step}: Val Acc = {val_metrics.get('accuracy', 0):.4f}")

    #             # TensorBoard 记录评估指标
    #             if self.tensorboard_writer is not None:
    #                 self.tensorboard_writer.add_scalar('Accuracy/val', val_metrics.get('accuracy', 0), self.global_step)
    #                 self.tensorboard_writer.add_scalar('Loss/val', val_metrics.get('loss', 0), self.global_step)

    #                 # 记录每个策略的准确率
    #                 strategy_acc = val_metrics.get('strategy_accuracy', {})
    #                 for strategy, acc in strategy_acc.items():
    #                     self.tensorboard_writer.add_scalar(f'Accuracy/val_{strategy}', acc, self.global_step)

    #                 # 记录路由分布
    #                 routing_dist = val_metrics.get('routing_distribution', {})
    #                 for strategy, ratio in routing_dist.items():
    #                     self.tensorboard_writer.add_scalar(f'RoutingDistribution/val_{strategy}', ratio, self.global_step)

    #         # 定期保存检查点
    #         if self.global_step % self.training_config.save_steps == 0:
    #             checkpoint_path = f"{self.output_dir}/checkpoint_step_{self.global_step}"
    #             self.save_checkpoint(checkpoint_path)
    #             print(f"Step {self.global_step}: Checkpoint saved to {checkpoint_path}")
        
    #     return {
    #         'loss': total_loss / num_batches if num_batches > 0 else 0.0
    #     }


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
            json.dump({
                'training_steps': self.global_step,
                'epochs': self.epoch + 1,
                'learning_rate': self.training_config.learning_rate,
            }, f, indent=2)
    
    def close(self):
        """关闭资源"""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            print("TensorBoard writer 已关闭")


# 注册到工厂
TrainableRouterFactory.register_trainer('classification')(ClassificationTrainer)
