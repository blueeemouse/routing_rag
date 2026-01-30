"""
DCTrainer实现

基于双对比学习的训练器
参考: LLMRouter/llmrouter/models/routerdc/trainer.py
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


class DCTrainer(BaseTrainer):
    """DCRouter训练器"""
    
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
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 学习率调度器
        self.scheduler = self._init_scheduler()
    
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
        if self.training_config.scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            total_steps = self.training_config.training_steps
            num_warmup_steps = int(0.1 * total_steps) if total_steps > 0 else 100
            
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps
            )
            return scheduler
        elif self.training_config.scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            total_steps = self.training_config.training_steps
            num_warmup_steps = int(0.1 * total_steps) if total_steps > 0 else 100
            
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
        计算损失

        Args:
            batch: 批次数据
                - input_ids: token ids (可选，仅用于非 SentenceTransformer)
                - attention_mask: 注意力掩码 (可选，仅用于非 SentenceTransformer)
                - scores: 策略分数, shape: (batch_size, num_strategies)
                - cluster_ids: cluster id (可选)
                - queries: 查询文本列表 (用于 SentenceTransformer)

        Returns:
            损失值
        """
        scores = batch['scores'].to(self.device)
        cluster_ids = batch.get('cluster_ids', None)

        if cluster_ids is not None:
            cluster_ids = cluster_ids.to(self.device)

        # 前向传播
        if hasattr(self.model, 'use_sentence_transformer') and self.model.use_sentence_transformer:
            # 使用sentence-transformers编码
            queries = batch['queries']
            query_emb = self.model.encode(queries)
        else:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            query_emb = self.model.forward(input_ids, attention_mask)
        
        # 获取策略embedding
        strategy_emb = self.model.get_strategy_embeddings()
        
        # 计算相似度
        similarity = self.model.compute_similarity(query_emb, strategy_emb)
        
        # 根据分数排序获取索引
        index_true = scores.argsort(dim=-1, descending=True)
        
        # 计算各项损失
        sample_llm_loss = self.model.compute_sample_llm_loss(
            similarity,
            index_true,
            top_k=self.training_config.top_k,
            last_k=self.training_config.last_k
        )
        
        total_loss = sample_llm_loss * self.training_config.sample_llm_loss_weight
        
        # Sample-Sample对比损失
        if self.training_config.sample_sample_loss_weight > 0 and cluster_ids is not None:
            sample_sample_loss = self.model.compute_sample_sample_loss(query_emb, cluster_ids)
            total_loss += sample_sample_loss * self.training_config.sample_sample_loss_weight
        
        # Cluster对比损失
        if self.training_config.cluster_loss_weight > 0 and cluster_ids is not None:
            cluster_loss = self.model.compute_cluster_loss(
                query_emb, 
                cluster_ids,
                self.data_config.num_clusters
            )
            total_loss += cluster_loss * self.training_config.cluster_loss_weight
        
        return total_loss
    
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
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        
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
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'step': self.global_step
            })
            
            # 定期评估
            if self.global_step % self.training_config.eval_steps == 0:
                val_metrics = self.evaluate(dataloader)
                print(f"Step {self.global_step}: Val Acc = {val_metrics.get('accuracy', 0):.4f}")
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型

        Args:
            dataloader: 数据加载器

        Returns:
            评估指标
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                scores = batch['scores'].to(self.device)
                queries = batch.get('queries', [])

                # 前向传播
                if hasattr(self.model, 'use_sentence_transformer') and self.model.use_sentence_transformer:
                    query_emb = self.model.encode(queries)
                else:
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
        
        # 计算每个策略的准确率
        strategy_accuracy = {}
        for i, strategy in enumerate(self.model.strategy_names):
            mask = [l == i for l in all_labels]
            if any(mask):
                strategy_correct = sum(p == l for p, l, m in zip(all_predictions, all_labels, mask) if m)
                strategy_total = sum(mask)
                strategy_accuracy[strategy] = strategy_correct / strategy_total if strategy_total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'strategy_accuracy': strategy_accuracy,
            'num_samples': len(all_labels),
        }
    
    def save_checkpoint(self, path: str):
        """
        保存检查点
        
        Args:
            path: 保存路径
        """
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
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
    
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
                'learning_rate': self.training_config.learning_rate,
            }, f, indent=2)


# 注册到工厂
TrainableRouterFactory.register_trainer('dc')(DCTrainer)
TrainableRouterFactory.register_trainer('dcrouter')(DCTrainer)
