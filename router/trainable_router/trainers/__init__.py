"""训练器模块"""

from .dc_trainer import DCTrainer
from .classification_trainer import ClassificationTrainer
from .statistical_trainer import StatisticalTrainer
from .soft_label_trainer import BinarySoftLabelTrainer, SoftLabelTrainer  # SoftLabelTrainer 为兼容别名
from .dpo_trainer import RouterDPOTrainer, ClassificationDPOTrainer

__all__ = [
    'DCTrainer', 
    'ClassificationTrainer', 
    'StatisticalTrainer',
    'BinarySoftLabelTrainer',
    'SoftLabelTrainer',  # 兼容旧名称
    'RouterDPOTrainer',
    'ClassificationDPOTrainer',
]
