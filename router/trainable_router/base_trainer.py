"""
可训练路由器训练器基类

定义所有训练器需要实现的通用接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
from tqdm import tqdm


class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(self, model, config, output_dir: str = "outputs"):
        """
        初始化
        
        Args:
            model: 模型
            config: 配置
            output_dir: 输出目录
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.global_step = 0
        self.epoch = 0
        
        # 设置设备
        self.device = torch.device(config.device if hasattr(config, 'device') and config.device != 'auto' else 'cpu')
        self.model.to(self.device)
    
    @abstractmethod
    def compute_loss(self, batch) -> torch.Tensor:
        """
        计算损失
        
        Args:
            batch: 批次数据
            
        Returns:
            损失值
        """
        pass
    
    @abstractmethod
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            训练指标
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            评估指标
        """
        pass
    
    def train(self, train_dataloader, val_dataloader=None, **kwargs) -> Dict[str, Any]:
        """
        训练主循环
        
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
        
        epochs = self.config.training.epochs if hasattr(self.config, 'training') else 10
        # 支持通过参数限制最大训练步数（max_steps），用于快速调试
        max_steps = kwargs.get('max_steps', None)
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_dataloader, max_steps=max_steps)
            # train_metrics['loss'] may be a single float or a list of floats
            loss_val = train_metrics.get('loss', [])
            if isinstance(loss_val, list):
                history['train_loss'].extend(loss_val)
            else:
                history['train_loss'].append(loss_val)
            
            # 验证
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                history['val_metrics'].append(val_metrics)
            
            # 保存检查点
            self.save_checkpoint(f"{self.output_dir}/checkpoint_epoch_{epoch}")

            # 如果设置了 max_steps 且已达到或超过该步数，提前结束训练
            if max_steps is not None and self.global_step >= max_steps:
                break
        
        return history
    
    def save_checkpoint(self, path: str):
        """
        保存检查点
        
        Args:
            path: 保存路径
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        torch.save(checkpoint, f"{path}/model.pt")
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(f"{path}/model.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
