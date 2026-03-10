#!/usr/bin/env python3
"""
DPO Router训练脚本

基于TRL库的DPOTrainer，使用MiniLM等小型模型进行路由器DPO训练。

用法:
    # 基础用法
    python train_dpo.py \
        --train_file HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_full.json \
        --val_file HotpotQA_train_data/label_analysis/dpo_data/dpo_preference_pairs_val.json \
        --output_dir outputs/dpo_router

    # 完整配置
    python train_dpo.py \
        --model_name sentence-transformers/all-MiniLM-L6-v2 \
        --train_file data/dpo_train.json \
        --val_file data/dpo_val.json \
        --output_dir outputs/dpo_experiment \
        --batch_size 16 \
        --epochs 3 \
        --learning_rate 1e-5 \
        --beta 0.1
"""

import os
import sys
import argparse
import json
import torch
import copy
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
)
from datasets import Dataset as HFDataset

# 添加项目路径（必须放在最前面）
router_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(router_dir)
if router_dir not in sys.path:
    sys.path.insert(0, router_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 添加项目根目录，用于导入interfaces等

# 现在可以安全导入
def get_classification_dpo_trainer():
    from trainable_router.trainers.dpo_trainer import ClassificationDPOTrainer
    return ClassificationDPOTrainer

def get_save_log_callback():
    from trainable_router.trainers.dpo_trainer import SaveLogCallback
    return SaveLogCallback


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DPO Router训练')
    
    # 模型配置
    parser.add_argument(
        '--model_name', '-m',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='预训练模型名称'
    )
    parser.add_argument(
        '--num_labels',
        type=int,
        default=2,
        help='分类标签数量（默认2：no_rag, naive_rag）'
    )
    parser.add_argument(
        '--strategy_names',
        type=str,
        nargs='+',
        default=['no_rag', 'naive_rag'],
        help='策略名称列表'
    )
    
    # 数据配置
    parser.add_argument(
        '--train_file', '-t',
        type=str,
        required=True,
        help='训练数据路径（DPO偏好对JSON格式）'
    )
    parser.add_argument(
        '--val_file', '-v',
        type=str,
        default=None,
        help='验证数据路径（可选）'
    )
    
    # 训练配置
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='训练批次大小'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=None,
        help='评估批次大小（默认等于batch_size）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='训练轮数'
    )
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=1e-5,
        help='学习率'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='权重衰减'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='最大序列长度'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.1,
        help='DPO温度参数（控制偏离reference模型的程度）'
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        default='sigmoid',
        choices=['sigmoid', 'hinge', 'ipo', 'kto'],
        help='DPO损失类型'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=100,
        help='预热步数'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=50,
        help='日志记录间隔'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=100,
        help='评估间隔'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=100,
        help='模型保存间隔'
    )
    parser.add_argument(
        '--save_total_limit',
        type=int,
        default=3,
        help='最多保存的检查点数量'
    )
    
    # 输出配置
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='outputs/dpo_router',
        help='输出目录'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    # 设备配置
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='训练设备'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='使用FP16混合精度'
    )
    parser.add_argument(
        '--bf16',
        action='store_true',
        default=False,
        help='使用BF16混合精度'
    )
    
    return parser.parse_args()


def load_dpo_data(file_path: str) -> list:
    """
    加载DPO偏好对数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式：列表或包含samples的字典
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'samples' in data:
        return data['samples']
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")


def prepare_hf_dataset(data: list) -> HFDataset:
    """
    将数据转换为HuggingFace Dataset格式
    
    Args:
        data: 原始数据列表
        
    Returns:
        HuggingFace Dataset
    """
    # 提取字段
    prompts = [item['prompt'] for item in data]
    chosens = [item['chosen'] for item in data]
    rejecteds = [item['rejected'] for item in data]
    
    return HFDataset.from_dict({
        'prompt': prompts,
        'chosen': chosens,
        'rejected': rejecteds,
    })


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 自动选择设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数: {torch.cuda.device_count()}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练参数
    args_dict = vars(args)
    with open(output_dir / 'training_args.json', 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("DPO Router训练配置")
    print("="*60)
    print(f"模型: {args.model_name}")
    print(f"策略: {args.strategy_names}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"DPO beta: {args.beta}")
    print(f"输出目录: {args.output_dir}")
    print("="*60 + "\n")
    
    # =============================
    # 1. 加载模型和Tokenizer
    # =============================
    print("加载模型和Tokenizer...")
    
    # 加载配置并指定num_labels
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    
    # 加载模型（自动添加分类头）
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 处理padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"模型加载完成: {args.model_name}")
    print(f"分类头输出维度: {args.num_labels}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # =============================
    # 2. 构建Reference模型
    # =============================
    print("\n构建Reference模型...")
    
    # 深拷贝并冻结参数
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print("Reference模型构建完成（已冻结参数）")
    
    # =============================
    # 3. 加载数据
    # =============================
    print("\n加载训练数据...")
    
    train_data = load_dpo_data(args.train_file)
    train_dataset = prepare_hf_dataset(train_data)
    print(f"训练样本数: {len(train_dataset)}")
    
    eval_dataset = None
    if args.val_file and Path(args.val_file).exists():
        print("加载验证数据...")
        val_data = load_dpo_data(args.val_file)
        eval_dataset = prepare_hf_dataset(val_data)
        print(f"验证样本数: {len(eval_dataset)}")
    else:
        print("未提供验证数据，将跳过评估")
    
    # =============================
    # 4. 配置DPO训练参数
    # =============================
    print("\n配置DPO训练...")
    
    try:
        from trl.trainer.dpo_config import DPOConfig
    except ImportError:
        print("错误: 未安装TRL库。请运行: pip install trl")
        return
    
    eval_batch_size = args.eval_batch_size or args.batch_size
    
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy='steps' if eval_dataset else 'no',
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        max_prompt_length=args.max_length,
        max_completion_length=1,  # 分类任务不使用
        beta=args.beta,
        loss_type=args.loss_type,
        generate_during_eval=False,
        remove_unused_columns=False,
        dataset_num_proc=None,
        report_to=['tensorboard'],
        logging_dir=str(output_dir / 'logs'),
        # 保存最佳模型（根据验证集准确率）
        load_best_model_at_end=True,
        metric_for_best_model='eval_rewards/accuracies',
        greater_is_better=True,
    )
    
    # =============================
    # 5. 初始化DPO Trainer
    # =============================
    print("初始化ClassificationDPOTrainer...")
    
    # 延迟导入
    ClassificationDPOTrainer = get_classification_dpo_trainer()
    SaveLogCallback = get_save_log_callback()
    
    trainer = ClassificationDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # 添加自定义日志回调
    trainer.add_callback(SaveLogCallback(log_file=str(output_dir / 'training_log.txt')))
    
    print("训练器初始化完成")
    
    # =============================
    # 6. 开始训练
    # =============================
    print("\n" + "="*60)
    print("开始DPO训练")
    print("="*60 + "\n")
    
    trainer.train()
    
    # =============================
    # 7. 保存最终模型和最佳模型
    # =============================
    print("\n保存最终模型...")
    
    final_model_path = output_dir / 'final'
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    # 保存策略名称映射
    with open(final_model_path / 'strategy_names.json', 'w', encoding='utf-8') as f:
        json.dump(args.strategy_names, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 最终模型已保存到: {final_model_path}")
    
    # 保存最佳模型（根据验证集准确率）
    if eval_dataset is not None and trainer.state.best_model_checkpoint is not None:
        best_checkpoint = trainer.state.best_model_checkpoint
        best_step = trainer.state.best_global_step
        best_metric = trainer.state.best_metric
        
        print(f"\n最佳模型信息:")
        print(f"  Checkpoint: {best_checkpoint}")
        print(f"  Step: {best_step}")
        print(f"  Val Accuracy: {best_metric:.4f}")
        
        # 创建 checkpoint_best_val 链接
        best_model_path = output_dir / 'checkpoint_best_val'
        
        import shutil
        if best_model_path.exists():
            if best_model_path.is_symlink():
                best_model_path.unlink()
            else:
                shutil.rmtree(best_model_path)
        
        # 复制最佳模型到 checkpoint_best_val
        shutil.copytree(best_checkpoint, best_model_path)
        
        # 保存最佳模型信息
        best_info = {
            'best_step': best_step,
            'best_metric': best_metric,
            'metric_name': 'eval_rewards/accuracies',
            'original_checkpoint': best_checkpoint,
            'model_path': str(best_model_path),
        }
        with open(output_dir / 'best_model_info.json', 'w', encoding='utf-8') as f:
            json.dump(best_info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 最佳模型已保存到: {best_model_path}")
    
    # 最终评估
    if eval_dataset is not None:
        print("\n进行最终评估...")
        eval_result = trainer.evaluate()
        print(f"评估结果: {eval_result}")
        
        # 保存评估结果
        with open(output_dir / 'eval_result.json', 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"输出目录: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
