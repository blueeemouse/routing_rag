"""
可训练路由器配置管理

支持多种模型和训练方式的配置
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    # Backbone配置
    backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 隐藏层大小
    hidden_size: int = 384
    
    # 策略名称列表（对应RAG实现）
    strategy_names: List[str] = field(default_factory=lambda: ["no_rag", "naive_rag", "graph_rag"])
    
    # 策略数量
    num_strategies: int = 3
    
    # 相似度计算方式: "cos" or "dot"
    similarity_function: str = "cos"
    
    # 温度参数（用于softmax）
    temperature: float = 1.0
    
    # 设备
    device: str = "cpu"
    
    # 预训练模型路径
    pretrained_model_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """训练配置"""
    # 批量大小
    batch_size: int = 32
    
    # 学习率
    learning_rate: float = 5.0e-5
    
    # 训练轮数
    epochs: int = 10
    
    # 最大序列长度
    max_length: int = 512
    
    # 训练步数
    training_steps: int = 0  # 0表示使用epochs
    
    # 评估步数
    eval_steps: int = 100
    
    # 保存步数
    save_steps: int = 500
    
    # 损失权重
    sample_llm_loss_weight: float = 1.0
    sample_sample_loss_weight: float = 0.1
    cluster_loss_weight: float = 1.0
    
    # Top-K / Last-K（用于对比学习）
    top_k: int = 3
    last_k: int = 3
    
    # 优化器配置
    optimizer_type: str = "adamw"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 学习率调度器
    scheduler_type: str = "linear"
    
    # 梯度裁剪
    max_grad_norm: float = 1.0
    
    # 混合精度训练
    use_amp: bool = False
    
    # 随机种子
    seed: int = 42


@dataclass
class DataConfig:
    """数据配置"""
    # 数据源类型: "hotpotqa" or "llm_judge"
    source: str = "hotpotqa"
    
    # 训练数据路径
    train_path: str = ""
    
    # 验证数据路径
    val_path: str = ""
    
    # 测试数据路径
    test_path: str = ""
    
    # Cluster数量（用于聚类对比学习）
    num_clusters: int = 50
    
    # 分数归一化
    normalize_scores: bool = True
    
    # 是否打乱数据
    shuffle: bool = True
    
    # 评分公式（支持自定义公式，如 "em * 0.3 + f1 * 0.7"）
    score_formula: str = "em"


@dataclass
class TrainableRouterConfig:
    """可训练路由器主配置"""
    # 模型类型: "dc", "knn", "mf", "rl"
    model_type: str = "dc"
    
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 数据配置
    data: DataConfig = field(default_factory=DataConfig)
    
    # 保存配置
    output_dir: str = "router_models"
    save_model_path: Optional[str] = None
    
    # 日志配置
    logging_dir: Optional[str] = None
    log_level: str = "info"
    
    # 设备
    device: str = "auto"  # auto, cpu, cuda
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainableRouterConfig":
        """从YAML文件加载配置"""
        import yaml
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        return cls.from_dict(raw_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainableRouterConfig":
        """从字典加载配置"""
        model_config = config_dict.get('model', {})
        training_config = config_dict.get('training', {})
        data_config = config_dict.get('data', {})
        
        model = ModelConfig(
            backbone_name=model_config.get('backbone_name', "sentence-transformers/all-MiniLM-L6-v2"),
            hidden_size=model_config.get('hidden_size', 384),
            strategy_names=model_config.get('strategy_names', ["no_rag", "naive_rag", "graph_rag"]),
            num_strategies=model_config.get('num_strategies', 3),
            similarity_function=model_config.get('similarity_function', 'cos'),
            temperature=model_config.get('temperature', 1.0),
            device=model_config.get('device', 'cpu'),
            pretrained_model_path=model_config.get('pretrained_model_path'),
        )
        
        training = TrainingConfig(
            batch_size=training_config.get('batch_size', 32),
            learning_rate=training_config.get('learning_rate', 5.0e-5),
            epochs=training_config.get('epochs', 10),
            max_length=training_config.get('max_length', 512),
            training_steps=training_config.get('training_steps', 0),
            eval_steps=training_config.get('eval_steps', 100),
            save_steps=training_config.get('save_steps', 500),
            sample_llm_loss_weight=training_config.get('sample_llm_loss_weight', 1.0),
            sample_sample_loss_weight=training_config.get('sample_sample_loss_weight', 0.1),
            cluster_loss_weight=training_config.get('cluster_loss_weight', 1.0),
            top_k=training_config.get('top_k', 3),
            last_k=training_config.get('last_k', 3),
        )
        
        data = DataConfig(
            source=data_config.get('source', 'hotpotqa'),
            train_path=data_config.get('train_path', ''),
            val_path=data_config.get('val_path', ''),
            test_path=data_config.get('test_path', ''),
            num_clusters=data_config.get('num_clusters', 50),
            normalize_scores=data_config.get('normalize_scores', True),
            shuffle=data_config.get('shuffle', True),
            score_formula=data_config.get('score_formula', 'em'),
        )
        
        return cls(
            model_type=config_dict.get('model_type', 'dc'),
            model=model,
            training=training,
            data=data,
            output_dir=config_dict.get('output_dir', 'router_models'),
            save_model_path=config_dict.get('save_model_path'),
            logging_dir=config_dict.get('logging_dir'),
            log_level=config_dict.get('log_level', 'info'),
            device=config_dict.get('device', 'auto'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type,
            'model': {
                'backbone_name': self.model.backbone_name,
                'hidden_size': self.model.hidden_size,
                'strategy_names': self.model.strategy_names,
                'num_strategies': self.model.num_strategies,
                'similarity_function': self.model.similarity_function,
                'temperature': self.model.temperature,
                'device': self.model.device,
                'pretrained_model_path': self.model.pretrained_model_path,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'epochs': self.training.epochs,
                'max_length': self.training.max_length,
                'training_steps': self.training.training_steps,
                'eval_steps': self.training.eval_steps,
                'save_steps': self.training.save_steps,
                'top_k': self.training.top_k,
                'last_k': self.training.last_k,
                'sample_llm_loss_weight': self.training.sample_llm_loss_weight,
                'sample_sample_loss_weight': self.training.sample_sample_loss_weight,
                'cluster_loss_weight': self.training.cluster_loss_weight,
            },
            'data': {
                'source': self.data.source,
                'train_path': self.data.train_path,
                'val_path': self.data.val_path,
                'test_path': self.data.test_path,
                'num_clusters': self.data.num_clusters,
                'normalize_scores': self.data.normalize_scores,
                'shuffle': self.data.shuffle,
                'score_formula': self.data.score_formula,
            },
            'output_dir': self.output_dir,
            'save_model_path': self.save_model_path,
            'logging_dir': self.logging_dir,
            'log_level': self.log_level,
            'device': self.device,
        }
