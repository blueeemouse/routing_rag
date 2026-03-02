#!/usr/bin/env python3
"""
模型对比评估脚本
比较两个模型在训练集和验证集上的：
1. Loss
2. 分类准确率
3. 混淆矩阵
4. 模型参数 L2 范数
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.trainable_router.config import TrainableRouterConfig
from router.trainable_router.factory import TrainableRouterFactory
from router.trainable_router.datasets.router_label_dataset import RouterLabelDataset
from router.trainable_router.datasets.hotpotqa_dataset import GenericRouterDataset


def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path, device='cuda'):
    """加载模型
    
    Returns:
        model: 加载的模型
        config: 配置对象（包含 model_type 字段）
    """
    print(f"加载模型: {model_path}")
    
    # 加载 hparams.json 获取完整配置
    hparams_path = os.path.join(os.path.dirname(model_path), 'hparams.json')
    if os.path.exists(hparams_path):
        with open(hparams_path, 'r') as f:
            hparams = json.load(f)
        print(f"  从 hparams.json 加载配置")
    else:
        hparams = None
        print(f"  警告：未找到 hparams.json，使用默认配置")
    
    # 创建配置
    config = TrainableRouterConfig.from_dict(hparams) if hparams else TrainableRouterConfig()
    
    # 使用工厂创建模型
    model = TrainableRouterFactory.create_model(config)
    model.to(device)
    
    # 更新模型内部的 device 属性（确保手工特征等能正确移动到 GPU）
    model.device = torch.device(device)
    
    # 加载权重
    checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=device, weights_only=False)
    
    # 检查 checkpoint 结构
    if 'model_state_dict' in checkpoint:
        # 标准 checkpoint 格式
        state_dict = checkpoint['model_state_dict']
    else:
        # 直接是 state_dict
        state_dict = checkpoint
    
    # strict=True 确保权重完全匹配，避免静默失败
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model, config


def compute_model_l2_norm(model):
    """计算模型所有参数的总 L2 范数"""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_model_l2_norm_from_params(model):
    """计算模型所有参数的总 L2 范数（不依赖梯度）"""
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def evaluate_model(model, dataloader, device='cuda', model_type='dc'):
    """评估模型，返回 loss、accuracy 和预测结果
    
    支持两种模型类型：
    1. dc (DCRouterModel): 基于相似度计算，forward(input_ids, attention_mask) -> embedding
    2. feature_fused (FeatureFusedRouterModel): 基于分类器，forward(input_ids, attention_mask, queries) -> logits
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0
    
    strategy_names = model.strategy_names
    is_similarity_based = (model_type == 'dc')
    
    print(f"  模型类型: {model_type} ({'相似度计算' if is_similarity_based else '直接分类'})")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch.get('scores', None)
            queries = batch.get('queries', None)
            
            if scores is not None:
                scores = scores.to(device)
                labels = scores.argmax(dim=-1)
            else:
                labels = None
            
            # 根据模型类型选择前向传播方式
            if is_similarity_based:
                # DCRouterModel: 基于相似度
                query_emb = model.forward(input_ids, attention_mask)
                strategy_emb = model.get_strategy_embeddings()
                logits = model.compute_similarity(query_emb, strategy_emb)
                
                temperature = model.temperature
                if temperature > 0:
                    logits = logits / temperature
            else:
                # FeatureFusedRouterModel: 直接分类
                if queries is None:
                    raise ValueError("FeatureFusedRouterModel 需要 queries 参数")
                logits = model.forward(input_ids, attention_mask, queries)
            
            # 计算 loss
            if scores is not None:
                loss_fn = nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * input_ids.size(0)
            
            # 预测
            predictions = logits.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
            num_samples += input_ids.size(0)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = (all_predictions == all_labels).mean()
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels
    }


def compute_confusion_matrix(predictions, labels, num_classes):
    """计算混淆矩阵"""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    for pred, label in zip(predictions, labels):
        confusion[label][pred] += 1
    return confusion


def print_confusion_matrix(cm, class_names):
    """打印混淆矩阵"""
    print("\n混淆矩阵:")
    print("=" * 60)
    
    # 打印表头
    header = "Predicted\\Actual".ljust(15)
    for name in class_names:
        header += name[:10].ljust(12)
    print(header)
    print("-" * 60)
    
    # 打印每一行
    for i, name in enumerate(class_names):
        row = name[:10].ljust(15)
        for j in range(len(class_names)):
            row += str(cm[i][j]).ljust(12)
        print(row)
    
    print("=" * 60)
    
    # 打印每个类的准确率
    print("\n各类别准确率:")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        print(f"  {name}: {correct}/{total} = {acc:.4f}")


def main():
    # 设置随机种子以确保可复现性
    set_seed(42)
    
    parser = argparse.ArgumentParser(description='模型对比评估')
    parser.add_argument('--model1', type=str, 
                        default='router_models/weight_decay_search_phaseA/wd_1e-2/checkpoint_step_50',
                        help='第一个模型路径')
    parser.add_argument('--model2', type=str, 
                        default='router_models/weight_decay_search_phaseB/wd_10.0/checkpoint_step_50',
                        help='第二个模型路径')
    parser.add_argument('--train_data', type=str,
                        default='HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json',
                        help='训练数据路径')
    parser.add_argument('--val_data', type=str,
                        default='evaluation_results/router_test_labels.json',
                        help='验证数据路径')
    parser.add_argument('--config', type=str,
                        default='config/train_classification_5000.yaml',
                        help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 先加载第一个模型来获取配置
    print("\n加载第一个模型配置...")
    model1, config = load_model(args.model1, device)
    model_type1 = config.model_type
    
    # 设置分词器 - 使用模型自带的 tokenizer
    tokenizer = model1.tokenizer if hasattr(model1, 'tokenizer') and model1.tokenizer else None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_name)
    print(f"使用分词器: {config.model.backbone_name}")
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = RouterLabelDataset(config)
    train_dataset.tokenizer = tokenizer  # 传递 tokenizer
    train_dataset.load_data(args.train_data)
    print(f"训练集大小: {len(train_dataset)}")
    
    # 创建 collate_fn - 与 train_router.py 保持一致
    def collate_fn(x):
        batch_data = {
            'scores': torch.tensor([item['scores'] for item in x], dtype=torch.float32),
            'cluster_ids': torch.tensor([item['cluster_id'] for item in x], dtype=torch.long),
            'queries': [item['queries'] for item in x],
        }
        # 统一使用transformers方式，始终添加分词数据
        if 'input_ids' in x[0]:
            # 数据已经分词好（如 GenericRouterDataset）
            batch_data['input_ids'] = torch.stack([item['input_ids'] for item in x])
            batch_data['attention_mask'] = torch.stack([item['attention_mask'] for item in x])
        elif tokenizer is not None:
            # 数据没有分词（如 RouterLabelDataset），需要实时分词
            encoded = tokenizer(
                [item['queries'] for item in x],
                padding=True,
                truncation=True,
                max_length=config.training.max_length,
                return_tensors='pt'
            )
            batch_data['input_ids'] = encoded['input_ids']
            batch_data['attention_mask'] = encoded['attention_mask']
        return batch_data
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 如果有验证集
    val_loader = None
    if args.val_data:
        val_dataset = RouterLabelDataset(config)
        val_dataset.tokenizer = tokenizer  # 传递 tokenizer
        val_dataset.load_data(args.val_data)
        print(f"验证集大小: {len(val_dataset)}")
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    # 加载第二个模型
    print("\n" + "=" * 60)
    model2, config2 = load_model(args.model2, device)
    model_type2 = config2.model_type
    
    # 策略名称
    strategy_names = config.model.strategy_names
    print(f"策略名称: {strategy_names}")
    
    # 评估函数
    def evaluate_single_model(model, name, loader, split_name, model_type):
        print(f"\n{'=' * 60}")
        print(f"评估 {name} ({split_name})")
        print("=" * 60)
        
        # 计算 L2 范数
        l2_norm = compute_model_l2_norm_from_params(model)
        print(f"模型参数 L2 范数: {l2_norm:.4f}")
        
        # 评估
        results = evaluate_model(model, loader, device, model_type)
        
        print(f"\n{split_name} Loss: {results['loss']:.4f}")
        print(f"{split_name} Accuracy: {results['accuracy']:.4f}")
        
        # 混淆矩阵
        cm = compute_confusion_matrix(results['predictions'], results['labels'], len(strategy_names))
        print_confusion_matrix(cm, strategy_names)
        
        return results, l2_norm
    
    # 评估 Model 1
    print("\n" + "=" * 80)
    print("模型 1")
    print("=" * 80)
    
    train_results_1, l2_1 = evaluate_single_model(model1, "Model 1", train_loader, "训练集", model_type1)
    if val_loader:
        val_results_1, _ = evaluate_single_model(model1, "Model 1", val_loader, "验证集", model_type1)
    
    # 评估 Model 2
    print("\n" + "=" * 80)
    print("模型 2")
    print("=" * 80)
    
    train_results_2, l2_2 = evaluate_single_model(model2, "Model 2", train_loader, "训练集", model_type2)
    if val_loader:
        val_results_2, _ = evaluate_single_model(model2, "Model 2", val_loader, "验证集", model_type2)
    
    # 对比总结
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"\n{'指标':<20} {'Model 1':<15} {'Model 2':<15} {'差异':<15}")
    print("-" * 60)
    print(f"{'训练集 Loss':<20} {train_results_1['loss']:<15.4f} {train_results_2['loss']:<15.4f} {train_results_2['loss']-train_results_1['loss']:<15.4f}")
    print(f"{'训练集 Acc':<20} {train_results_1['accuracy']:<15.4f} {train_results_2['accuracy']:<15.4f} {train_results_2['accuracy']-train_results_1['accuracy']:<15.4f}")
    print(f"{'参数 L2 范数':<20} {l2_1:<15.4f} {l2_2:<15.4f} {l2_2-l2_1:<15.4f}")
    
    if val_loader:
        print(f"{'验证集 Loss':<20} {val_results_1['loss']:<15.4f} {val_results_2['loss']:<15.4f} {val_results_2['loss']-val_results_1['loss']:<15.4f}")
        print(f"{'验证集 Acc':<20} {val_results_1['accuracy']:<15.4f} {val_results_2['accuracy']:<15.4f} {val_results_2['accuracy']-val_results_1['accuracy']:<15.4f}")
    
    print("\n分析:")
    if l2_1 > l2_2:
        print(f"  - Model 1 的参数范数更大 ({l2_1:.4f} vs {l2_2:.4f})")
    else:
        print(f"  - Model 2 的参数范数更大 ({l2_2:.4f} vs {l2_1:.4f})")
    
    if train_results_1['accuracy'] > train_results_2['accuracy']:
        print(f"  - Model 1 在训练集上准确率更高")
    else:
        print(f"  - Model 2 在训练集上准确率更高")


if __name__ == '__main__':
    main()
