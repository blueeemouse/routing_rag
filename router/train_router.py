#!/usr/bin/env python3
"""
路由器训练脚本

用法:
    python train_router.py --config config.yaml
    python train_router.py --model_type dc --train_data data/train.jsonl --output_dir outputs
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import random
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainable_router.config import TrainableRouterConfig
from trainable_router.factory import TrainableRouterFactory
from trainable_router.utils.logger import setup_logging, get_logger


def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练路由器模型')
    
    # 配置文件
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    # 模型配置
    parser.add_argument('--model_type', type=str, default=None, help='模型类型 (dc, knn, mf)')
    parser.add_argument('--backbone', type=str, default=None, 
                        help='Backbone模型名称')
    
    # 数据配置
    parser.add_argument('--train_data', type=str, default=None, help='训练数据路径')
    parser.add_argument('--val_data', type=str, default=None, help='验证数据路径')
    parser.add_argument('--test_data', type=str, default=None, help='测试数据路径')
    parser.add_argument('--data_source', type=str, default=None, help='数据源类型')
    
    # 内部表征配置
    parser.add_argument('--representation_type', type=str, default=None, 
                        help='表征类型 (shallow_mean, deep_last_token, concat_all 等)')
    parser.add_argument('--representation_dim', type=int, default=None,
                        help='表征维度 (单一2048, concat_deep=4096, concat_all=12288)')
    
    # 混合表征融合配置
    parser.add_argument('--representation_dir', type=str, default=None,
                        help='内部表征目录路径')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='标签JSON文件路径')
    parser.add_argument('--fusion_type', type=str, default=None,
                        help='融合方式 (cross_attn, bidirectional_cross_attn)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='冻结 MiniLM backbone')
    parser.add_argument('--freeze_internal_rep_proj', action='store_true',
                        help='冻结内部表征投影层')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_length', type=int, default=None, help='最大序列长度')
    parser.add_argument('--eval_steps', type=int, default=None, help='评估步数')
    parser.add_argument('--save_steps', type=int, default=None, help='保存步数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--trainer_type', type=str, default=None, help='训练器类型 (dc, classification, feature_fused)')
    
    # 损失配置
    parser.add_argument('--top_k', type=int, default=None, help='正样本数量')
    parser.add_argument('--last_k', type=int, default=None, help='负样本数量')
    parser.add_argument('--sample_llm_loss_weight', type=float, default=None, help='Sample-LLM损失权重')
    parser.add_argument('--cluster_loss_weight', type=float, default=None, help='Cluster损失权重')
    parser.add_argument('--num_clusters', type=int, default=None, help='Cluster数量')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='类别权重，格式: "no_rag=3.0,naive_rag=1.0"')
    parser.add_argument('--tie_weight', type=float, default=None,
                        help='Tie样本权重（默认非tie样本权重为1.0）')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--save_model_path', type=str, default=None, help='模型保存路径')
    
    # 设备配置
    parser.add_argument('--device', type=str, default=None, help='设备 (auto, cpu, cuda)')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--max_steps', type=int, default=None, help='训练的最大step数（0表示不限制）')
    
    # 模型参数
    parser.add_argument('--temperature', type=float, default=None, help='温度参数')
    
    # Debug配置
    parser.add_argument('--overfit_single_batch', action='store_true', help='过拟合单个batch')
    parser.add_argument('--fast_dev_steps', type=int, default=None, help='快速开发模式下的训练步数')
    
    # 正则化配置
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减（L2正则化）')
    
    # 继续训练
    parser.add_argument('--resume', type=str, default='', help='从检查点恢复训练')
    
    # 脚本路径（用于记录脚本内容）
    parser.add_argument('--script_path', type=str, default='', help='PowerShell脚本路径（自动记录脚本内容）')
    
    return parser.parse_args()


def create_config_from_args(args) -> TrainableRouterConfig:
    """
    从命令行参数创建配置（智能合并）
    
    优先级：命令行参数（非None）> 配置文件（如果指定）> 默认值
    
    Args:
        args: 命令行参数
        
    Returns:
        TrainableRouterConfig
    """
    from router.trainable_router.config import ModelConfig, TrainingConfig, DataConfig
    
    # 先创建基础配置（如果有配置文件则加载）
    if args.config:
        config = TrainableRouterConfig.from_yaml(args.config)
    else:
        # 创建空配置，使用Config的默认值
        config = TrainableRouterConfig()
    
    # 用非None的命令行参数覆盖配置
    
    # 模型配置
    if args.model_type is not None:
        config.model_type = args.model_type
    if args.backbone is not None:
        config.model.backbone_name = args.backbone
    if args.device is not None:
        config.model.device = args.device
        config.device = args.device
    if args.temperature is not None:
        config.model.temperature = args.temperature
    
    # 训练配置
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = float(args.learning_rate)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.max_length is not None:
        config.training.max_length = args.max_length
    if args.eval_steps is not None:
        config.training.eval_steps = args.eval_steps
    if args.save_steps is not None:
        config.training.save_steps = args.save_steps
    if args.seed is not None:
        config.training.seed = args.seed
    if args.trainer_type is not None:
        config.training.trainer_type = args.trainer_type
    if args.top_k is not None:
        config.training.top_k = args.top_k
    if args.last_k is not None:
        config.training.last_k = args.last_k
    if args.sample_llm_loss_weight is not None:
        config.training.sample_llm_loss_weight = args.sample_llm_loss_weight
    if args.cluster_loss_weight is not None:
        config.training.cluster_loss_weight = args.cluster_loss_weight
    
    # 【新增】处理类别权重
    if args.class_weights is not None:
        strategy_names = config.model.strategy_names
        weights = parse_class_weights(args.class_weights, strategy_names)
        config.training.class_weights = weights
    
    if args.use_amp:
        config.training.use_amp = args.use_amp
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    
    # 【新增】处理weight_decay
    if args.weight_decay is not None:
        config.training.optimizer_kwargs['weight_decay'] = args.weight_decay
    
    # Debug配置
    # 如果启用overfit_single_batch，则会对同一个batch反复进行训练
    if args.overfit_single_batch:
        config.training.overfit_single_batch = args.overfit_single_batch
    # fast_dev_steps控制的是，在overfit_single_batch模式下，每个epoch训练的步数（所以即使是过拟合模式下，epoch数还是照常）
    if args.fast_dev_steps is not None:
        config.training.fast_dev_steps = args.fast_dev_steps
    
    # 数据配置
    if args.data_source is not None:
        config.data.source = args.data_source
    if args.train_data is not None:
        config.data.train_path = args.train_data
    if args.val_data is not None:
        config.data.val_path = args.val_data
    if args.test_data is not None:
        config.data.test_path = args.test_data
    if args.num_clusters is not None:
        config.data.num_clusters = args.num_clusters
    
    # 内部表征配置
    if args.representation_type is not None:
        config.data.representation_type = args.representation_type
    if args.representation_dim is not None:
        config.model.representation_dim = args.representation_dim
    
    # 混合表征融合配置
    if args.representation_dir is not None:
        config.data.representation_dir = args.representation_dir
    if args.labels_path is not None:
        config.data.labels_path = args.labels_path
    if args.fusion_type is not None:
        config.model.fusion_type = args.fusion_type
    if args.freeze_backbone:
        config.model.freeze_backbone = True
    if args.freeze_internal_rep_proj:
        config.model.freeze_internal_rep_proj = True
    
    # 输出配置
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.save_model_path is not None:
        config.save_model_path = args.save_model_path
    else:
        config.save_model_path = config.output_dir
    
    return config


def parse_class_weights(weights_str: str, strategy_names: list) -> dict:
    """
    解析类别权重字符串
    
    Args:
        weights_str: 权重字符串，如 "no_rag=3.0,naive_rag=1.0"
        strategy_names: 策略名称列表，如 ["no_rag", "naive_rag"]
        
    Returns:
        权重字典，未指定的策略默认权重为1.0
    """
    if not weights_str:
        return {name: 1.0 for name in strategy_names}
    
    weights = {}
    pairs = weights_str.split(',')
    for pair in pairs:
        if '=' in pair:
            strategy, weight = pair.split('=', 1)
            strategy = strategy.strip()
            weight = float(weight.strip())
            weights[strategy] = weight
    
    # 为未指定的策略设置默认权重1.0
    for name in strategy_names:
        if name not in weights:
            weights[name] = 1.0
    
    return weights


def config_to_dict(config: Any) -> Dict:
    """
    将config对象递归转换为字典（只保留基本数据类型）
    
    Args:
        config: 配置对象
        
    Returns:
        包含所有配置信息的字典
    """
    result = {}
    
    if hasattr(config, '__dict__'):
        for key, value in config.__dict__.items():
            if key.startswith('_'):
                continue
            
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            elif isinstance(value, (list, tuple)):
                processed_list = []
                for item in value:
                    if isinstance(item, (str, int, float, bool)):
                        processed_list.append(item)
                    elif hasattr(item, '__dict__'):
                        processed_list.append(config_to_dict(item))
                    else:
                        processed_list.append(str(item))
                result[key] = processed_list
            elif isinstance(value, dict):
                processed_dict = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool)):
                        processed_dict[k] = v
                    elif hasattr(v, '__dict__'):
                        processed_dict[k] = config_to_dict(v)
                    else:
                        processed_dict[k] = str(v)
                result[key] = processed_dict
            elif hasattr(value, '__dict__'):
                result[key] = config_to_dict(value)
            else:
                result[key] = str(value)
    
    return result


def save_config_to_json(config: TrainableRouterConfig, output_dir: str) -> str:
    """
    保存完整config为JSON文件
    
    Args:
        config: 配置对象
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    # 转换为字典
    config_dict = config_to_dict(config)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存到文件
    hparams_path = output_path / "hparams.json"
    with open(hparams_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    return str(hparams_path)


def generate_exp_name(config: TrainableRouterConfig) -> str:
    """
    根据配置自动生成实验名称
    
    规则：exp_<mode>_<temp>_<backbone>_<trainer_type>_<timestamp>
    示例：exp_norm_t0.05_minilm_classification_0207_1230
    """
    parts = ["exp"]
    
    # 1. 模式
    mode = "overfit" if config.training.overfit_single_batch else "norm"
    if config.training.fast_dev_steps:
        mode += f"_dev{config.training.fast_dev_steps}"
    parts.append(mode)
    
    # 2. 温度
    temp = config.model.temperature
    parts.append(f"t{temp}")
    
    # 3. Backbone（取最后一部分）
    backbone = config.model.backbone_name.split('/')[-1]
    parts.append(backbone)
    
    # 4. Trainer类型
    trainer_type = config.training.trainer_type
    parts.append(trainer_type)
    
    # 5. 时间戳
    timestamp = datetime.now().strftime("%m%d_%H%M")
    parts.append(timestamp)
    
    return "_".join(parts)


def main():
    """主函数"""
    # 设置随机种子以确保可复现性
    set_seed(42)
    
    args = parse_args()
    
    # 记录完整命令行
    command = ' '.join(sys.argv)

    # 验证参数
    if not args.config and not args.train_data:
        print("错误：必须提供 --config 或 --train_data 参数")
        parser.print_help()
        sys.exit(1)

    # 加载配置（智能合并：命令行参数覆盖配置文件）
    config = create_config_from_args(args)
    
    # 设置输出目录：优先级 命令行 > 配置文件 > 自动生成
    if not args.output_dir:
        # 命令行未指定，检查配置文件是否指定了 output_dir
        if config.output_dir:
            # 配置文件有指定，使用它
            args.output_dir = config.output_dir
        else:
            # 配置文件也没有，自动生成
            exp_name = generate_exp_name(config)
            args.output_dir = f"router_models/experiments/{exp_name}"
            config.output_dir = args.output_dir
    
    # 初始化日志系统 (必须在加载配置之后)
    log_dir = f"{config.output_dir}/logs"

    logger = setup_logging(
        log_dir=log_dir,
        log_level=config.log_level
    )
    logger_instance = logger.get_logger("train_router")
    
    # 记录训练开始信息
    logger.log_training_start(command, vars(args), config)
    
    # 保存完整配置到JSON
    hparams_path = save_config_to_json(config, config.output_dir)
    logger_instance.info(f"完整配置已保存到: {hparams_path}")

    # 设置随机种子
    seed = config.training.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger_instance.info(f"PyTorch 随机种子已设置为: {seed}")

    # 设置设备
    device = config.device or config.model.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.model.device = device
        config.device = device
        logger_instance.info(f"自动选择设备: {device}")
    else:
        logger_instance.info(f"使用指定设备: {device}")
    
    # 验证参数合理性
    logger_instance.info("验证参数...")
    # 验证top_k和last_k（这里的逻辑很简单，只是让top_k和last_k不要超过候选策略数）
    num_strategies = config.model.num_strategies
    top_k = config.training.top_k
    last_k = config.training.last_k
    
    if top_k >= num_strategies:
        logger_instance.warning(f"top_k ({top_k}) >= num_strategies ({num_strategies})，可能导致错误")
        logger_instance.info(f"自动调整top_k为: {num_strategies - 1}")
        config.training.top_k = num_strategies - 1
    
    if last_k >= num_strategies:
        logger_instance.warning(f"last_k ({last_k}) >= num_strategies ({num_strategies})，可能导致错误")
        logger_instance.info(f"自动调整last_k为: {num_strategies - 1}")
        config.training.last_k = num_strategies - 1
    
    logger_instance.info("参数验证完成")
    
    
    # 创建数据集
    logger_instance.info("加载数据集...")
    
    # 标记是否跳过后续的 load_data 调用
    skip_train_load = False
    skip_val_load = False
    
    # 优先根据 config.data.source 选择数据集类型
    source_type = getattr(config.data, 'source', '').lower()
    
    if source_type == 'hybrid_representation_fusion' or source_type == 'hybrid_rep_fusion':
        # 混合表征融合数据集（内部表征 + 语义表征）
        logger_instance.info(f"使用HybridRepresentationFusionDataset (source={source_type})")
        from trainable_router.datasets.hybrid_representation_fusion_dataset import HybridRepresentationFusionDataset
        representation_type = getattr(config.data, 'representation_type', 'deep_last_token')
        train_dataset = HybridRepresentationFusionDataset(
            config, 
            representation_type=representation_type
        )
        # 设置数据路径（从配置读取）并加载
        train_dataset.representation_dir = config.data.representation_dir
        train_dataset.labels_path = config.data.labels_path
        train_dataset.load_data(config.data.representation_dir)
        skip_train_load = True
    elif source_type == 'internal_representation' or source_type == 'internal_rep':
        # 内部表征数据集（预提取的 LLM 表征）
        logger_instance.info(f"使用InternalRepresentationDataset (source={source_type})")
        from trainable_router.datasets.internal_representation_dataset import InternalRepresentationDataset
        representation_type = getattr(config.data, 'representation_type', 'deep_last_token')
        train_dataset = InternalRepresentationDataset(
            config, 
            representation_type=representation_type
        )
    elif source_type == 'decision_router':
        # 决策式路由数据集（预测Q和cost）
        logger_instance.info(f"使用DecisionRouterDataset (source={source_type})")
        from trainable_router.datasets.decision_router_dataset import DecisionRouterDataset
        train_dataset = DecisionRouterDataset(config)
    elif source_type == 'soft_label':
        # 软标签数据集（纯统计特征）
        logger_instance.info(f"使用SoftLabelRouterDataset (source={source_type})")
        from trainable_router.datasets.soft_label_dataset import SoftLabelRouterDataset
        train_dataset = SoftLabelRouterDataset(config)
    elif source_type == 'fusion_soft_label':
        # 融合模型软标签数据集（语义+统计特征）
        logger_instance.info(f"使用FusionSoftLabelDataset (source={source_type})")
        from trainable_router.datasets.fusion_soft_label_dataset import FusionSoftLabelDataset
        train_dataset = FusionSoftLabelDataset(config)
    elif ('router_test_labels' in config.data.train_path or 
          'router_labels' in config.data.train_path or
          'all_labels' in config.data.train_path or
          'curriculum_stage2' in config.data.train_path):
        
        # 根据tie_weight参数选择数据集
        if args.tie_weight is not None:
            logger_instance.info(f"使用WeightedRouterLabelDataset（tie_weight={args.tie_weight}）")
            from trainable_router.datasets.weighted_router_label_dataset import WeightedRouterLabelDataset
            train_dataset = WeightedRouterLabelDataset(config, tie_weight=args.tie_weight)
        else:
            logger_instance.info("使用RouterLabelDataset")
            from trainable_router.datasets.router_label_dataset import RouterLabelDataset
            train_dataset = RouterLabelDataset(config)
    else:
        from trainable_router.datasets.hotpotqa_dataset import GenericRouterDataset
        train_dataset = GenericRouterDataset(config)
    
    # 加载训练数据（除非已经在数据集创建时加载）
    if not skip_train_load:
        train_dataset.load_data(config.data.train_path)

    print('length of train_dataset:', len(train_dataset))

    val_dataset = None
    # 对于 hybrid_representation_fusion，检查是否有验证表征目录
    val_representation_dir = getattr(config.data, 'val_representation_dir', None)
    
    if config.data.val_path or val_representation_dir:
        # 根据 source 类型选择验证数据集
        if source_type == 'hybrid_representation_fusion' or source_type == 'hybrid_rep_fusion':
            # 混合表征融合数据集
            logger_instance.info(f"使用HybridRepresentationFusionDataset加载验证集 (source={source_type})")
            from trainable_router.datasets.hybrid_representation_fusion_dataset import HybridRepresentationFusionDataset
            representation_type = getattr(config.data, 'representation_type', 'deep_last_token')
            val_dataset = HybridRepresentationFusionDataset(
                config,
                representation_type=representation_type
            )
            # 使用验证表征目录（如果指定），强制从 metadata 加载标签
            if val_representation_dir:
                val_dataset.load_data(val_representation_dir, from_metadata=True)
                skip_val_load = True
        elif source_type == 'internal_representation' or source_type == 'internal_rep':
            # 内部表征数据集
            logger_instance.info(f"使用InternalRepresentationDataset加载验证集 (source={source_type})")
            from trainable_router.datasets.internal_representation_dataset import InternalRepresentationDataset
            representation_type = getattr(config.data, 'representation_type', 'deep_last_token')
            val_dataset = InternalRepresentationDataset(
                config,
                representation_type=representation_type
            )
        elif source_type == 'decision_router':
            # 决策式路由使用DecisionRouterDataset
            logger_instance.info(f"使用DecisionRouterDataset加载验证集 (source={source_type})")
            from trainable_router.datasets.decision_router_dataset import DecisionRouterDataset
            val_dataset = DecisionRouterDataset(config)
        elif source_type == 'fusion_soft_label':
            # 融合模型使用 FusionSoftLabelDataset，支持验证集无 soft_label
            logger_instance.info(f"使用FusionSoftLabelDataset加载验证集 (source={source_type})")
            from trainable_router.datasets.fusion_soft_label_dataset import FusionSoftLabelDataset
            val_dataset = FusionSoftLabelDataset(config)
        elif source_type == 'soft_label':
            logger_instance.info(f"使用SoftLabelRouterDataset加载验证集 (source={source_type})")
            from trainable_router.datasets.soft_label_dataset import SoftLabelRouterDataset
            val_dataset = SoftLabelRouterDataset(config)
        elif config.data.val_path and ('router_test_labels' in config.data.val_path or 'router_labels' in config.data.val_path):
            logger_instance.info("检测到路由标签格式，使用RouterLabelDataset")
            from trainable_router.datasets.router_label_dataset import RouterLabelDataset
            val_dataset = RouterLabelDataset(config)
        else:
            val_dataset = GenericRouterDataset(config)

        # 加载验证数据（除非已经在数据集创建时加载）
        if not skip_val_load and config.data.val_path and config.data.val_path != '-':
            val_dataset.load_data(config.data.val_path)
        logger_instance.info(f"验证数据集已加载")
    else:
        logger_instance.info("未提供验证数据集，将跳过验证")

    # 先创建模型，以便collate_fn能根据模型属性做出一致判断
    logger_instance.info("创建模型...")
    # 根据config里的model_type参数创建router模型
    # 对于内部表征模型或混合表征融合模型，传递 representation_dim
    if (source_type == 'internal_representation' or source_type == 'internal_rep' or
        source_type == 'hybrid_representation_fusion' or source_type == 'hybrid_rep_fusion'):
        representation_dim = getattr(config.model, 'representation_dim', 2048)
        model = TrainableRouterFactory.create_model(config, representation_dim=representation_dim)
    else:
        model = TrainableRouterFactory.create_model(config)
    logger_instance.info(f"模型创建成功: {config.model_type}")

    # 将model的tokenizer传递给数据集（如果有的话），以便dataset能生成input_ids
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        train_dataset.tokenizer = model.tokenizer
        if val_dataset:
            val_dataset.tokenizer = model.tokenizer

    

    # 创建collate_fn
    def collate_fn(x):
        # 处理混合表征融合数据集（HybridRepresentationFusionDataset）
        if 'representation' in x[0] and 'input_ids' in x[0]:
            batch_data = {
                'representation': torch.stack([item['representation'] for item in x]),
                'input_ids': torch.stack([item['input_ids'] for item in x]),
                'attention_mask': torch.stack([item['attention_mask'] for item in x]),
                'label': torch.stack([item['label'] for item in x]),
                'queries': [item['query'] for item in x],
            }
            return batch_data
        
        # 处理内部表征数据集（InternalRepresentationDataset）
        if 'representation' in x[0]:
            batch_data = {
                'representation': torch.stack([item['representation'] for item in x]),
                'label': torch.stack([item['label'] for item in x]),
                'query': [item['query'] for item in x],
            }
            return batch_data
        
        # 处理决策式路由数据集（DecisionRouterDataset）- 新格式
        if 'Q' in x[0] and 'costs' in x[0]:
            batch_data = {
                'input_ids': torch.stack([item['input_ids'] for item in x]),
                'attention_mask': torch.stack([item['attention_mask'] for item in x]),
                'queries': [item['queries'] for item in x],
                'Q': torch.tensor([item['Q'] for item in x], dtype=torch.float32),
                'costs': torch.tensor([item['costs'] for item in x], dtype=torch.float32),
                'label': torch.tensor([item['label'] for item in x], dtype=torch.long),
                'cluster_id': torch.tensor([item['cluster_id'] for item in x], dtype=torch.long),
            }
            return batch_data
        
        # 处理软标签数据集（如 FusionSoftLabelDataset）
        if 'soft_label' in x[0]:
            soft_labels_tensor = torch.tensor([item['soft_label'] for item in x], dtype=torch.float32)
            batch_data = {
                'scores': soft_labels_tensor,  # 兼容其他训练器
                'soft_label': soft_labels_tensor,  # FusionSoftLabelTrainer 需要
                'cluster_ids': torch.tensor([item['cluster_id'] for item in x], dtype=torch.long),
                'queries': [item['queries'] for item in x],
            }
            
            # 软标签数据集已经分词好
            batch_data['input_ids'] = torch.stack([item['input_ids'] for item in x])
            batch_data['attention_mask'] = torch.stack([item['attention_mask'] for item in x])
            
            # 如果有硬标签也加上
            if 'label' in x[0]:
                batch_data['label'] = torch.tensor([item['label'] for item in x], dtype=torch.long)
            else:
                # 从 soft_label 推断硬标签（取概率最大的）
                soft_labels_np = soft_labels_tensor.numpy()
                inferred_labels = soft_labels_np.argmax(axis=-1)
                batch_data['label'] = torch.tensor(inferred_labels, dtype=torch.long)
            
            return batch_data
        
        # 普通数据集处理
        batch_data = {
            'scores': torch.tensor([item['scores'] for item in x], dtype=torch.float32),
            'cluster_ids': torch.tensor([item['cluster_id'] for item in x], dtype=torch.long),
            'queries': [item['queries'] for item in x],
        }
        
        # 如果item中有sample_weight，添加到batch
        if 'sample_weight' in x[0]:
            batch_data['sample_weights'] = torch.tensor(
                [item['sample_weight'] for item in x], 
                dtype=torch.float32
            )
        
        # 统一使用transformers方式，始终添加分词数据
        if 'input_ids' in x[0]:
            # 数据已经分词好（如 GenericRouterDataset）
            batch_data['input_ids'] = torch.stack([item['input_ids'] for item in x])
            batch_data['attention_mask'] = torch.stack([item['attention_mask'] for item in x])
        elif hasattr(model, 'tokenizer') and model.tokenizer is not None:
            # 数据没有分词（如 RouterLabelDataset），需要实时分词
            encoded = model.tokenizer(
                [item['queries'] for item in x],
                padding=True,
                truncation=True,
                max_length=config.training.max_length,
                return_tensors='pt'
            )
            batch_data['input_ids'] = encoded['input_ids']
            batch_data['attention_mask'] = encoded['attention_mask']
        return batch_data

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=0,
        collate_fn=collate_fn
    )
    # 是默认的true
    # print('train loader is shuffle?: ', config.data.shuffle)

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
    
    # 计算训练步数
    total_steps = len(train_loader) * config.training.epochs
    config.training.training_steps = total_steps
    
    # 记录数据信息
    strategy_stats = train_dataset._get_coverage_stats() if hasattr(train_dataset, '_get_coverage_stats') else None
    logger.log_data_info(
        train_samples=len(train_dataset),
        val_samples=len(val_dataset) if val_dataset else 0,
        strategy_stats=strategy_stats
    )
    
    logger_instance.info(f"总训练步数: {total_steps}")
    if total_steps > 0:
        logger_instance.info(f"评估间隔: 每 {config.training.eval_steps} 步")
        logger_instance.info(f"保存间隔: 每 {config.training.save_steps} 步")
    
    # 创建训练器
    trainer = TrainableRouterFactory.create_trainer(model, config, config.output_dir, logger=logger_instance)
    print('trainer:', trainer)
    # TensorBoard 信息
    tensorboard_dir = f"{config.output_dir}/tensorboard"
    logger_instance.info(f"TensorBoard 日志目录: {tensorboard_dir}")
    logger_instance.info(f"启动 TensorBoard 命令: tensorboard --logdir {tensorboard_dir}")

    # 判断是否要继续训练
    if args.resume:
        logger_instance.info(f"从检查点恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    else:
        logger_instance.info("开始新训练")
    
    # 训练
    logger_instance.info("开始训练...")
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
    history = trainer.train(train_loader, val_loader, max_steps=max_steps)
    
    # 保存最终模型
    final_model_path = f"{config.output_dir}/final"
    logger_instance.info(f"保存最终模型到: {final_model_path}")
    trainer.save_final_model(final_model_path)
    
    # 评估
    if val_loader:
        logger_instance.info("进行最终评估...")
        metrics = trainer.evaluate(val_loader)
        logger.log_evaluation(metrics)

    # 记录训练结束
    total_epochs = config.training.epochs
    total_steps = trainer.global_step if hasattr(trainer, 'global_step') else 0
    logger.log_training_end(total_epochs, total_steps)

    # 关闭 TensorBoard writer
    if hasattr(trainer, 'close'):
        trainer.close()
        logger_instance.info("训练资源已清理")


if __name__ == '__main__':
    main()
