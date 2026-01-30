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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练路由器模型')
    
    # 配置文件
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    # 模型配置
    parser.add_argument('--model_type', type=str, default='dc', help='模型类型 (dc, knn, mf)')
    parser.add_argument('--backbone', type=str, default='sentence-transformers/all-MiniLM-L6-v2', 
                        help='Backbone模型名称')
    
    # 数据配置
    parser.add_argument('--train_data', type=str, default='', help='训练数据路径')
    parser.add_argument('--val_data', type=str, default='', help='验证数据路径')
    parser.add_argument('--test_data', type=str, default='', help='测试数据路径')
    parser.add_argument('--data_source', type=str, default='hotpotqa', help='数据源类型')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=5.0e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--eval_steps', type=int, default=100, help='评估步数')
    parser.add_argument('--save_steps', type=int, default=500, help='保存步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 损失配置
    parser.add_argument('--top_k', type=int, default=3, help='正样本数量')
    parser.add_argument('--last_k', type=int, default=3, help='负样本数量')
    parser.add_argument('--sample_llm_loss_weight', type=float, default=1.0, help='Sample-LLM损失权重')
    parser.add_argument('--cluster_loss_weight', type=float, default=1.0, help='Cluster损失权重')
    parser.add_argument('--num_clusters', type=int, default=50, help='Cluster数量')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='router_models', help='输出目录')
    parser.add_argument('--save_model_path', type=str, default='', help='模型保存路径')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto, cpu, cuda)')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--max_steps', type=int, default=0, help='训练的最大step数（0表示不限制）')
    
    # 继续训练
    parser.add_argument('--resume', type=str, default='', help='从检查点恢复训练')
    
    return parser.parse_args()


def create_config_from_args(args) -> TrainableRouterConfig:
    """
    从命令行参数创建配置
    
    Args:
        args: 命令行参数
        
    Returns:
        TrainableRouterConfig
    """
    from router.trainable_router.config import ModelConfig, TrainingConfig, DataConfig
    
    model_config = ModelConfig(
        backbone_name=args.backbone,
        strategy_names=['no_rag', 'naive_rag', 'graph_rag'],
        num_strategies=3,
        similarity_function='cos',
        device=args.device,
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_length=args.max_length,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        top_k=args.top_k,
        last_k=args.last_k,
        sample_llm_loss_weight=args.sample_llm_loss_weight,
        cluster_loss_weight=args.cluster_loss_weight,
        use_amp=args.use_amp,
        seed=args.seed,
    )
    
    data_config = DataConfig(
        source=args.data_source,
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        num_clusters=args.num_clusters,
        shuffle=True,
    )
    
    return TrainableRouterConfig(
        model_type=args.model_type,
        model=model_config,
        training=training_config,
        data=data_config,
        output_dir=args.output_dir,
        save_model_path=args.save_model_path or args.output_dir,
        device=args.device,
    )


def main():
    """主函数"""
    args = parse_args()

    # 验证参数
    if not args.config and not args.train_data:
        print("错误：必须提供 --config 或 --train_data 参数")
        parser.print_help()
        sys.exit(1)

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载配置
    if args.config:
        config = TrainableRouterConfig.from_yaml(args.config)
    else:
        config = create_config_from_args(args)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.model.device = device
        config.device = device
    
    print("=" * 60)
    print("路由器训练")
    print("=" * 60)
    print(f"模型类型: {config.model_type}")
    print(f"Backbone: {config.model.backbone_name}")
    print(f"设备: {config.device}")
    print(f"训练数据: {config.data.train_path}")
    print(f"输出目录: {config.output_dir}")
    print("=" * 60)
    
    # 创建数据集
    print("\n加载数据集...")
    from trainable_router.datasets.hotpotqa_dataset import GenericRouterDataset
    
    train_dataset = GenericRouterDataset(config)
    train_dataset.load_data(config.data.train_path)
    
    val_dataset = None
    if config.data.val_path:
        val_dataset = GenericRouterDataset(config)
        val_dataset.load_data(config.data.val_path)

    # 先创建模型，以便 collate_fn 能根据 model.use_sentence_transformer 做出一致判断
    print("\n创建模型...")
    model = TrainableRouterFactory.create_model(config)

    # 根据模型属性判断是否使用 sentence-transformer
    use_sentence_transformer = getattr(model, 'use_sentence_transformer', False)

    # 将 model 的 tokenizer 传递给数据集（如果有的话），以便 dataset 能生成 input_ids
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        train_dataset.tokenizer = model.tokenizer
        if val_dataset:
            val_dataset.tokenizer = model.tokenizer

    # 定义辅助函数：判断是否使用 sentence-transformer（与 dc_model 中的逻辑保持一致）
    def _is_sentence_transformer(backbone_name: str) -> bool:
        """判断是否使用 sentence-transformers"""
        return 'sentence-transformers' in backbone_name or 'all-MiniLM' in backbone_name
    

    # 创建 collate_fn
    def collate_fn(x):
        batch_data = {
            'scores': torch.tensor([item['scores'] for item in x], dtype=torch.float32),
            'cluster_ids': torch.tensor([item['cluster_id'] for item in x], dtype=torch.long),
            'queries': [item['queries'] for item in x],
        }
        # 只有在使用非 SentenceTransformer 时才添加分词数据
        if not use_sentence_transformer and 'input_ids' in x[0]:
            batch_data['input_ids'] = torch.stack([item['input_ids'] for item in x])
            batch_data['attention_mask'] = torch.stack([item['attention_mask'] for item in x])
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
    
    print(f"训练样本数: {len(train_dataset)}")
    if val_dataset:
        print(f"验证样本数: {len(val_dataset)}")
    
    # 计算训练步数
    total_steps = len(train_loader) * config.training.epochs
    config.training.training_steps = total_steps
    
    # 创建训练器
    trainer = TrainableRouterFactory.create_trainer(model, config, config.output_dir)
    
    # 继续训练
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 训练
    print("\n开始训练...")
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
    history = trainer.train(train_loader, val_loader, max_steps=max_steps)
    
    # 保存最终模型
    print(f"\n保存模型到: {config.output_dir}/final")
    trainer.save_final_model(f"{config.output_dir}/final")
    
    # 评估
    if val_loader:
        print("\n最终评估...")
        metrics = trainer.evaluate(val_loader)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"损失: {metrics['loss']:.4f}")
    
    print("\n训练完成!")
    print(f"模型保存路径: {config.output_dir}/final")


if __name__ == '__main__':
    main()
