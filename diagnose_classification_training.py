"""
诊断分类训练问题
检查数据、模型参数、梯度等
"""

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'router'))

from trainable_router.config import TrainableRouterConfig
from trainable_router.factory import TrainableRouterFactory
from trainable_router.datasets.router_label_dataset import RouterLabelDataset

def diagnose_data(data_path, config):
    """诊断数据"""
    print("=" * 80)
    print("数据诊断")
    print("=" * 80)
    
    # 加载数据集
    dataset = RouterLabelDataset(config)
    dataset.load_data(data_path)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 检查标签分布
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        scores = torch.tensor(item['scores'])
        label = scores.argmax().item()
        labels.append(label)
    
    label_counts = Counter(labels)
    print(f"标签分布:")
    for label, count in label_counts.items():
        strategy_name = dataset.model.strategy_names[label] if hasattr(dataset, 'model') else f"索引{label}"
        print(f"  {strategy_name}: {count} ({count/len(labels):.2%})")
    
    # 检查前几个样本
    # print("\n前5个样本:")
    # for i in range(min(5, len(dataset))):
    #     item = dataset[i]
    #     print(f"  样本 {i}:")
    #     print(f"    question: {item.get('queries', [''])[0][:80]}...")
    #     print(f"    scores: {item['scores']}")
    #     print(f"    label: {torch.tensor(item['scores']).argmax().item()}")
    
    return dataset, labels

def diagnose_model(model, dataloader, device):
    """诊断模型"""
    print("\n" + "=" * 80)
    print("模型诊断")
    print("=" * 80)
    
    # 获取一个batch
    batch = next(iter(dataloader))
    
    print(f"Batch keys: {list(batch.keys())}")
    
    # 检查scores
    scores = batch['scores']
    print('scores:', scores)
    print('scores type:', type(scores))
    print(f"Scores length: {len(scores)}")
    # print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    # print(f"Scores sample:\n{scores[:5]}")
    
    # 检查labels
    # labels = scores.argmax(dim=-1)
    labels = np.argmax(scores, axis=0)
    print(f"\nLabels: {labels[:10]}...")
    print(f"Label distribution: {[np.sum(labels==i) for i in np.unique(labels)]}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        print('queries:', batch['queries'])
        queries = batch['queries']
        query_emb = model.encode(queries).to(device)
        
        strategy_emb = model.get_strategy_embeddings()
        logits = model.compute_similarity(query_emb, strategy_emb)
        
        print(f"\nLogits shape: {logits.shape}")
        print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"Logits sample:\n{logits[:5]}")
        
        # 预测
        predictions = logits.argmax(dim=-1)
        print(f"\nPredictions: {predictions[:10]}...")
        
        # 准确率
        labels = torch.tensor(labels).to(device)
        print('predictions shape:', predictions.shape)
        print('lables shape:', labels.shape)
        correct = (predictions == labels).float().mean()
        print(f"Batch accuracy: {correct:.4f}")
        
        # 计算损失
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        print(f"Batch loss: {loss:.4f}")
    
    return batch, logits, labels, loss

def diagnose_parameters(model):
    """诊断模型参数"""
    print("\n" + "=" * 80)
    print("模型参数诊断")
    print("=" * 80)
    
    # 检查策略embeddings
    strategy_embs = model.get_strategy_embeddings()
    print(f"Strategy embeddings shape: {strategy_embs.shape}")
    print(f"Strategy embeddings norm: {torch.norm(strategy_embs, dim=1)}")
    
    # 检查 backbone 参数
    print("\nBackbone参数检查:")
    for name, param in model.named_parameters():
        if 'backbone' in name or 'strategy_embeddings' in name:
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}, "
                  f"device={param.device}, norm={torch.norm(param):.4f}")
    
    # 检查是否有参数需要梯度
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print('trainable parameter sample:', trainable_params[0])
    print(f"\n可训练参数数量: {len(trainable_params)}")
    print(f"总参数数量: {len(list(model.parameters()))}")

def check_gradient_flow(model, dataloader, device):
    """检查梯度流动"""
    print("\n" + "=" * 80)
    print("梯度流动检查")
    print("=" * 80)
    
    # 训练模式
    model.train()
    
    # 获取一个batch
    batch = next(iter(dataloader))
    print('batch:', batch)
    
    # 前向传播
    queries = batch['queries']
    query_emb = model.encode(queries)

    strategy_emb = model.get_strategy_embeddings()
    logits = model.compute_similarity(query_emb, strategy_emb)
    
    # 计算损失
    scores = batch['scores']
    labels = torch.tensor(np.argmax(scores, axis=0)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print("梯度检查:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            print(f"  {name}: 梯度范数={grad_norm:.6f}")
        else:
            print(f"  {name}: 无梯度")
    
    # 清理梯度
    model.zero_grad()

def main():
    """主函数"""
    # 配置
    config_path = r"D:\Develop\all_RAG\routing_rag\config\train_classification_5000.yaml"
    config = TrainableRouterConfig.from_yaml(config_path)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 数据诊断
    data_path = r"D:\Develop\all_RAG\routing_rag\HotpotQA_train_data\label_analysis\all_labels_converted.json"
    dataset, labels = diagnose_data(data_path, config)
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # 为了可重复性，先不shuffle
        num_workers=0,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    )
    
    # 模型
    model = TrainableRouterFactory.create_model(config)
    print('original model device:', model.device)
    model.to(device)
    model.device = device
    print('current model device:', model.device)
    
    # 模型参数诊断
    diagnose_parameters(model)
    
    # 数据+模型诊断
    diagnose_model(model, train_loader, device)
    
    # 梯度检查
    check_gradient_flow(model, train_loader, device)
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
