"""训练器模块"""

from .dc_trainer import DCTrainer
from .classification_trainer import ClassificationTrainer
from .statistical_trainer import StatisticalTrainer

__all__ = ['DCTrainer', 'ClassificationTrainer', 'StatisticalTrainer']
