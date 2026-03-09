"""
可训练路由器模块

支持多种训练方式和模型的路由器系统

主要组件：
- models: 模型定义（DC, kNN, MF, RL）
- trainers: 训练器（对比学习, 强化学习, 简单规则）
- datasets: 数据集（HotpotQA, LLM-as-a-judge）
- routers: 推理路由器
- data_utils: 数据预处理工具

使用示例：
    from router.trainable_router.factory import TrainableRouterFactory
    
    # 创建数据集
    dataset = TrainableRouterFactory.create_dataset(config)
    
    # 创建模型
    model = TrainableRouterFactory.create_model(config)
    
    # 创建训练器
    trainer = TrainableRouterFactory.create_trainer(model, config)
    
    # 训练
    trainer.train(dataset)
"""

from .config import TrainableRouterConfig, ModelConfig, TrainingConfig, DataConfig
from .factory import TrainableRouterFactory
from .base_model import BaseRouterModel
from .base_trainer import BaseTrainer
from .base_dataset import BaseRouterDataset
from .data_utils import DataAdapter, TrainingItem, ScoreComputer

# 数据集
from .datasets.hotpotqa_dataset import GenericRouterDataset, RouterDataLoader

# 模型
from .models.dc_model import DCRouterModel
from .models.feature_fused_model import FeatureFusedRouterModel
from .models.gated_fusion_model import GatedFusionRouterModel

# 训练器
from .trainers.dc_trainer import DCTrainer
from .trainers.feature_fused_trainer import FeatureFusedTrainer

# 路由器
from .routers.dc_router import DCRouter


__all__ = [
    # 配置
    'TrainableRouterConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    
    # 工厂
    'TrainableRouterFactory',
    
    # 基类
    'BaseRouterModel',
    'BaseTrainer',
    'BaseRouterDataset',
    
    # 工具
    'DataAdapter',
    'TrainingItem',
    'ScoreComputer',
    
    # 数据集
    'GenericRouterDataset',
    'RouterDataLoader',
    
    # 模型
    'DCRouterModel',
    'FeatureFusedRouterModel',
    'GatedFusionRouterModel',
    
    # 训练器
    'DCTrainer',
    'FeatureFusedTrainer',
    
    # 路由器
    'DCRouter',
]