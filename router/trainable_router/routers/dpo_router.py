"""
DPO Router 推理类

用于加载 DPO 训练好的分类模型进行推理。
"""

import os
import json
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 使用绝对导入
from interfaces.router_interface import RouterInterface


class DPORouter(RouterInterface):
    """
    DPO Router 推理类
    
    加载 DPO 训练好的序列分类模型进行路由决策。
    """
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        初始化
        
        Args:
            model_path: 训练好的模型路径（包含 config.json, model.safetensors, tokenizer 等）
            device: 设备（'cuda', 'cpu', 或 'auto'）
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        # 加载策略名称映射
        self._load_strategy_names()
        
        # 加载 tokenizer 和模型
        self._init_model()
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """设置设备"""
        if device is None or device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_strategy_names(self):
        """加载策略名称映射"""
        strategy_names_path = os.path.join(self.model_path, 'strategy_names.json')
        config_path = os.path.join(self.model_path, 'config.json')
        
        if os.path.exists(strategy_names_path):
            with open(strategy_names_path, 'r', encoding='utf-8') as f:
                self.strategy_names = json.load(f)
        elif os.path.exists(config_path):
            # 尝试从 config.json 读取
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 检查是否有 id2label
                if 'id2label' in config:
                    id2label = config['id2label']
                    self.strategy_names = [id2label[str(i)] for i in range(len(id2label))]
                else:
                    # 默认策略名称
                    num_labels = config.get('num_labels', 2)
                    self.strategy_names = [f'strategy_{i}' for i in range(num_labels)]
        else:
            # 默认二分类
            self.strategy_names = ['no_rag', 'naive_rag']
        
        self.num_strategies = len(self.strategy_names)
        print(f"策略名称: {self.strategy_names}")
    
    def _init_model(self):
        """初始化模型和 tokenizer"""
        print(f"加载 DPO 模型: {self.model_path}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 获取模型配置
        self.num_labels = self.model.config.num_labels
        self.hidden_size = self.model.config.hidden_size
        
        print(f"模型加载完成: {self.model.config.model_type}")
        print(f"  分类数: {self.num_labels}")
        print(f"  隐藏层维度: {self.hidden_size}")
    
    def route(self, query: str) -> str:
        """
        对查询进行路由决策
        
        Args:
            query: 输入查询
            
        Returns:
            选择的策略名称
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 获取预测类别
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        # 返回策略名称
        return self.strategy_names[predicted_class]
    
    def route_with_scores(self, query: str) -> Dict[str, Any]:
        """
        对查询进行路由决策，返回详细分数
        
        Args:
            query: 输入查询
            
        Returns:
            包含策略名称、分数、概率的字典
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 计算概率
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # 获取预测类别
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        # 构建结果
        scores = {
            'selected_strategy': self.strategy_names[predicted_class],
            'selected_idx': predicted_class,
            'logits': logits.squeeze(0).cpu().numpy().tolist(),
            'probabilities': {name: float(probs[i]) for i, name in enumerate(self.strategy_names)},
            'all_scores': {name: float(logits.squeeze(0).cpu().numpy()[i]) for i, name in enumerate(self.strategy_names)}
        }
        
        return scores
    
    def route_batch(self, queries: List[str]) -> List[str]:
        """
        批量路由决策
        
        Args:
            queries: 查询列表
            
        Returns:
            策略名称列表
        """
        # Tokenize
        inputs = self.tokenizer(
            queries,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 获取预测类别
            predicted_classes = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # 返回策略名称列表
        return [self.strategy_names[idx] for idx in predicted_classes]
