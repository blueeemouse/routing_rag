"""
可训练路由器工厂

提供统一的接口创建模型、训练器、数据集
"""

from typing import Dict, Type, Optional
import importlib

from .config import TrainableRouterConfig
from .base_model import BaseRouterModel
from .base_trainer import BaseTrainer
from .base_dataset import BaseRouterDataset


class TrainableRouterFactory:
    """可训练路由器工厂"""
    
    # 模型注册表
    _models: Dict[str, Type[BaseRouterModel]] = {}
    
    # 训练器注册表
    _trainers: Dict[str, Type[BaseTrainer]] = {}
    
    # 数据集注册表
    _datasets: Dict[str, Type[BaseRouterDataset]] = {}
    
    @classmethod
    def register_model(cls, model_type: str):
        """
        注册模型类
        
        Args:
            model_type: 模型类型标识
        """
        def decorator(model_class):
            cls._models[model_type] = model_class
            return model_class
        return decorator
    
    @classmethod
    def register_trainer(cls, trainer_type: str):
        """
        注册训练器类
        
        Args:
            trainer_type: 训练器类型标识
        """
        def decorator(trainer_class):
            cls._trainers[trainer_type] = trainer_class
            return trainer_class
        return decorator
    
    @classmethod
    def register_dataset(cls, source_type: str):
        """
        注册数据集类
        
        Args:
            source_type: 数据源类型标识
        """
        def decorator(dataset_class):
            cls._datasets[source_type] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def create_model(cls, config: TrainableRouterConfig, **kwargs) -> BaseRouterModel:
        """
        创建模型
        
        Args:
            config: 配置
            **kwargs: 额外参数
            
        Returns:
            模型实例
        """
        model_type = config.model_type.lower()
        
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._models.keys())}")
        
        return cls._models[model_type](config.model, **kwargs)
    
    @classmethod
    def create_trainer(cls, model: BaseRouterModel, config: TrainableRouterConfig, output_dir: str = "outputs") -> BaseTrainer:
        """
        创建训练器
        
        Args:
            model: 模型
            config: 配置
            output_dir: 输出目录
            
        Returns:
            训练器实例
        """
        trainer_type = config.model_type.lower()
        
        if trainer_type not in cls._trainers:
            raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {list(cls._trainers.keys())}")
        
        return cls._trainers[trainer_type](model, config, output_dir)
    
    @classmethod
    def create_dataset(cls, config: TrainableRouterConfig, tokenizer=None, split: str = "train") -> BaseRouterDataset:
        """
        创建数据集
        
        Args:
            config: 配置
            tokenizer: 分词器
            split: 数据集划分
            
        Returns:
            数据集实例
        """
        source_type = config.data.source.lower()
        
        if source_type not in cls._datasets:
            raise ValueError(f"Unknown data source: {source_type}. Available: {list(cls._datasets.keys())}")
        
        # 根据split选择数据路径
        if split == "train":
            data_path = config.data.train_path
        elif split == "val":
            data_path = config.data.val_path
        else:
            data_path = config.data.test_path
        
        dataset = cls._datasets[source_type](config, tokenizer)
        dataset.load_data(data_path)
        
        return dataset
    
    @classmethod
    def get_available_models(cls) -> list:
        """获取可用的模型类型"""
        return list(cls._models.keys())
    
    @classmethod
    def get_available_trainers(cls) -> list:
        """获取可用的训练器类型"""
        return list(cls._trainers.keys())
    
    @classmethod
    def get_available_datasets(cls) -> list:
        """获取可用的数据集类型"""
        return list(cls._datasets.keys())
