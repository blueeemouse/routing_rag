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
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainable_router.config import TrainableRouterConfig
from trainable_router.factory import TrainableRouterFactory
from trainable_router.utils.logger import setup_logging, get_logger


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
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max_length', type=int, default=None, help='最大序列长度')
    parser.add_argument('--eval_steps', type=int, default=None, help='评估步数')
    parser.add_argument('--save_steps', type=int, default=None, help='保存步数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    # 损失配置
    parser.add_argument('--top_k', type=int, default=None, help='正样本数量')
    parser.add_argument('--last_k', type=int, default=None, help='负样本数量')
    parser.add_argument('--sample_llm_loss_weight', type=float, default=None, help='Sample-LLM损失权重')
    parser.add_argument('--cluster_loss_weight', type=float, default=None, help='Cluster损失权重')
    parser.add_argument('--num_clusters', type=int, default=None, help='Cluster数量')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--save_model_path', type=str, default=None, help='模型保存路径')
    
    # 设备配置
    parser.add_argument('--device', type=str, default=None, help='设备 (auto, cpu, cuda)')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--max_steps', type=int, default=None, help='训练的最大step数（0表示不限制）')
    
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
    if args.top_k is not None:
        config.training.top_k = args.top_k
    if args.last_k is not None:
        config.training.last_k = args.last_k
    if args.sample_llm_loss_weight is not None:
        config.training.sample_llm_loss_weight = args.sample_llm_loss_weight
    if args.cluster_loss_weight is not None:
        config.training.cluster_loss_weight = args.cluster_loss_weight
    if args.use_amp:
        config.training.use_amp = args.use_amp
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    
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
    
    # 输出配置
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.save_model_path is not None:
        config.save_model_path = args.save_model_path
    else:
        config.save_model_path = config.output_dir
    
    return config


def main():
    """主函数"""
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
    
    # 初始化日志系统 (必须在加载配置之后)
    logger = setup_logging(
        log_dir=config.logging_dir or f"{config.output_dir}/logs",
        log_level=config.log_level
    )
    logger_instance = logger.get_logger("train_router")
    
    # 记录训练开始信息
    logger.log_training_start(command, vars(args), config)

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
    # from trainable_router.datasets.hotpotqa_dataset import GenericRouterDataset

    # train_dataset = GenericRouterDataset(config)
    # 根据数据路径判断数据集类型
    # if 'router_test_labels' in config.data.train_path or 'router_labels' in config.data.train_path:

    if ('router_test_labels' in config.data.train_path or 
        'router_labels' in config.data.train_path or
        'all_labels' in config.data.train_path):
        logger_instance.info("检测到路由标签格式，使用RouterLabelDataset")
        from trainable_router.datasets.router_label_dataset import RouterLabelDataset
        train_dataset = RouterLabelDataset(config)
    else:
        from trainable_router.datasets.hotpotqa_dataset import GenericRouterDataset
        train_dataset = GenericRouterDataset(config)
    train_dataset.load_data(config.data.train_path)

    val_dataset = None
    if config.data.val_path:
        # 检查是否是路由标签格式
        if 'router_test_labels' in config.data.val_path or 'router_labels' in config.data.val_path:
            logger_instance.info("检测到路由标签格式，使用RouterLabelDataset")
            from trainable_router.datasets.router_label_dataset import RouterLabelDataset
            val_dataset = RouterLabelDataset(config)
        else:
            val_dataset = GenericRouterDataset(config)

        val_dataset.load_data(config.data.val_path)
        logger_instance.info(f"验证数据集已加载: {config.data.val_path}")
    else:
        logger_instance.info("未提供验证数据集，将跳过验证")

    # 先创建模型，以便 collate_fn 能根据 model.use_sentence_transformer 做出一致判断
    logger_instance.info("创建模型...")
    # 根据config里的model_type参数创建router模型
    model = TrainableRouterFactory.create_model(config)
    logger_instance.info(f"模型创建成功: {config.model_type}")

    # 根据模型属性判断是否使用 sentence-transformer
    use_sentence_transformer = getattr(model, 'use_sentence_transformer', False)

    # 将 model 的 tokenizer 传递给数据集（如果有的话），以便 dataset 能生成 input_ids
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        train_dataset.tokenizer = model.tokenizer
        if val_dataset:
            val_dataset.tokenizer = model.tokenizer

    

    # 创建 collate_fn
    def collate_fn(x):
        batch_data = {
            'scores': torch.tensor([item['scores'] for item in x], dtype=torch.float32),
            'cluster_ids': torch.tensor([item['cluster_id'] for item in x], dtype=torch.long),
            'queries': [item['queries'] for item in x],
        }
        # 只有在使用非 SentenceTransformer 时才添加分词数据
        if not use_sentence_transformer:
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
    trainer = TrainableRouterFactory.create_trainer(model, config, config.output_dir)
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
