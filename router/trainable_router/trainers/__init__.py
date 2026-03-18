"""训练器模块"""

from .dc_trainer import DCTrainer
from .classification_trainer import ClassificationTrainer
from .statistical_trainer import StatisticalTrainer
from .soft_label_trainer import BinarySoftLabelTrainer, SoftLabelTrainer  # SoftLabelTrainer 为兼容别名
from .fusion_soft_label_trainer import FusionSoftLabelTrainer
from .dpo_trainer import RouterDPOTrainer, ClassificationDPOTrainer
from .knn_trainer import KNNTrainer
from .decision_router_trainer import DecisionRouterTrainer  # 新增：决策式路由训练器

__all__ = [
    'DCTrainer', 
    'ClassificationTrainer', 
    'StatisticalTrainer',
    'BinarySoftLabelTrainer',
    'SoftLabelTrainer',  # 兼容旧名称
    'FusionSoftLabelTrainer',
    'RouterDPOTrainer',
    'ClassificationDPOTrainer',
    'KNNTrainer',
    'DecisionRouterTrainer',  # 新增：决策式路由训练器
]
