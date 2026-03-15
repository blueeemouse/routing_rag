"""
路由器数据集抽象基类

提供统一的数据加载、策略映射和特征处理接口。
所有具体数据集类应继承此类。
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

import torch
from torch.utils.data import Dataset


class BaseRouterDataset(Dataset, ABC):
    """路由器数据集抽象基类
    
    核心职责：
    1. 统一加载 JSON 格式数据文件
    2. 统一策略名称到索引的映射
    3. 支持可选的语义特征（tokenize）
    4. 定义子类必须实现的抽象接口
    
    设计原则：
    - 统计特征由模型现算，数据集只返回原始 queries 文本
    - 语义特征（input_ids/attention_mask）通过配置开关控制
    - 标签格式（硬/软）由子类具体实现
    
    Attributes:
        config: 数据集配置对象
        split: 数据分割（"train"/"val"/"test"）
        raw_samples: 原始样本列表
        strategy_names: 策略名称列表
        strategy_to_idx: 策略名称到索引的映射
        num_strategies: 策略数量
        use_semantic: 是否使用语义特征
        tokenizer: 分词器（使用语义特征时需要）
        max_length: 最大序列长度
    """
    
    def __init__(self, config, split: str = "train"):
        """
        初始化基类
        
        Args:
            config: 配置对象，需包含 model.strategy_names, data, training.max_length 等
            split: 数据分割，"train"/"val"/"test"
        """
        super().__init__()
        
        self.config = config
        self.split = split
        
        # 获取数据路径
        self.data_path = self._get_data_path()
        
        # 初始化样本存储
        self.data: List[Dict[str, Any]] = []
        
        # 策略映射（从配置读取）
        self.strategy_names: List[str] = config.model.strategy_names
        self.strategy_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.strategy_names)
        }
        self.num_strategies: int = len(self.strategy_names)
        
        # 语义特征配置
        self.use_semantic: bool = getattr(config.data, 'use_semantic', False)
        self.tokenizer: Optional[Any] = None
        # max_length: None 表示使用 tokenizer 默认的 model_max_length
        self.max_length: Optional[int] = getattr(config.training, 'max_length', None)
        
        # 加载数据
        if self.data_path:
            self.load_data()
    
    def _get_data_path(self) -> Optional[str]:
        """根据 split 获取数据路径
        
        Returns:
            数据文件路径，如果未配置则返回 None
        """
        if self.split == "train":
            return getattr(self.config.data, 'train_path', None)
        elif self.split == "val":
            return getattr(self.config.data, 'val_path', None)
        elif self.split == "test":
            return getattr(self.config.data, 'test_path', None)
        return None
    
    def load_data(self) -> None:
        """加载 JSON 数据文件
        
        从 self.data_path 加载数据，存储到 self.raw_samples。
        子类可在加载后调用 _process_samples() 进行额外处理。
        """
        if not self.data_path or not os.path.exists(self.data_path):
            print(f"警告: 数据文件不存在或路径未配置: {self.data_path}")
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
        self.data = data.get('samples', [])
        print(f"成功加载 {len(self.data)} 条样本 from {self.data_path}")
            
            # 子类可以重写此方法进行额外处理
            self._process_samples()
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析错误 {self.data_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"加载数据失败 {self.data_path}: {e}")
    
    def _process_samples(self) -> None:
        """处理原始样本（子类可重写）
        
        默认空实现，子类可在此进行：
        - 数据验证
        - 字段转换
        - 统计信息计算
        """
        pass
    
    def set_tokenizer(self, tokenizer) -> None:
        """设置分词器（用于语义特征）
        
        必须在 use_semantic=True 且需要获取样本前调用。
        
        Args:
            tokenizer: HuggingFace transformers 格式的分词器
        """
        self.tokenizer = tokenizer
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """对文本进行 tokenize
        
        Args:
            text: 输入文本
            
        Returns:
            包含 input_ids 和 attention_mask 的字典
            
        Raises:
            ValueError: 如果 tokenizer 未设置
        """
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer 未设置。请先调用 set_tokenizer()，"
                "或将 use_semantic 设为 False。"
            )
        
        # max_length=None 时使用 tokenizer 默认的 model_max_length
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    @abstractmethod
    def get_label(self, sample: Dict[str, Any]) -> Union[int, List[float], torch.Tensor]:
        """从样本中解析标签（子类必须实现）
        
        Args:
            sample: 单个原始样本字典
            
        Returns:
            - 硬标签：类别索引（int）或 one-hot 向量（List[float]/Tensor）
            - 软标签：概率分布向量（List[float]/Tensor）
        """
        pass
    
    def get_strategy_index(self, strategy_name: str) -> int:
        """获取策略名称对应的索引
        
        Args:
            strategy_name: 策略名称（如 "no_rag", "naive_rag"）
            
        Returns:
            策略索引，如果未找到则返回 0
        """
        return self.strategy_to_idx.get(strategy_name, 0)
    
    def get_strategy_name(self, strategy_idx: int) -> str:
        """获取策略索引对应的名称
        
        Args:
            strategy_idx: 策略索引
            
        Returns:
            策略名称，如果索引越界则返回第一个策略名
        """
        if 0 <= strategy_idx < len(self.strategy_names):
            return self.strategy_names[strategy_idx]
        return self.strategy_names[0] if self.strategy_names else ""
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本（子类必须实现）
        
        子类实现应返回包含以下字段的字典：
        - queries: str, 原始问题文本（必需，用于统计特征）
        - label: 标签（硬标签索引或软标签向量）
        - input_ids: torch.Tensor (optional), 语义特征
        - attention_mask: torch.Tensor (optional), 语义特征
        - cluster_id: int (optional), 聚类标识
        
        Args:
            idx: 样本索引
            
        Returns:
            样本字典
        """
        pass
    
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.data)
    
    def get_sample_info(self, idx: int = 0) -> Dict[str, Any]:
        """获取样本信息（用于调试）
        
        Args:
            idx: 样本索引
            
        Returns:
            样本信息字典
        """
        if not self.data:
            return {"error": "无数据"}
        
        sample = self.data[idx]
        return {
            "index": idx,
            "total_samples": len(self.data),
            "has_question": "question" in sample,
            "has_label": "optimal_strategy" in sample or "label" in sample,
            "has_soft_label": "soft_label" in sample or "soft_label_vector" in sample,
            "sample_keys": list(sample.keys())
        }
    
    def validate_data(self) -> Dict[str, Any]:
        """验证数据完整性
        
        Returns:
            验证结果字典
        """
        if not self.data:
            return {"valid": False, "error": "无数据"}
        
        issues = []
        valid_count = 0
        
        for i, sample in enumerate(self.data[:100]):  # 只检查前100条
            # 检查必需字段
            if "question" not in sample:
                issues.append(f"样本 {i}: 缺少 question 字段")
                continue
            
            if not sample.get("question"):
                issues.append(f"样本 {i}: question 为空")
                continue
            
            valid_count += 1
        
        return {
            "valid": len(issues) == 0,
            "total": len(self.data),
            "checked": min(100, len(self.data)),
            "valid_in_checked": valid_count,
            "issues": issues[:10]  # 只返回前10个问题
        }
