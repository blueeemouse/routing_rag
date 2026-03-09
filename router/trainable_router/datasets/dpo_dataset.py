"""
DPO (Direct Preference Optimization) 偏好对数据集

用于加载DPO训练所需的偏好对数据格式:
[
    {
        "prompt": "问题文本",
        "chosen": 0,           // 偏好的策略索引
        "rejected": 1,         // 不喜欢的策略索引
        "question_id": "q_0",
        "optimal_strategy": "no_rag",
        "strategy_names": ["no_rag", "naive_rag"]
    }
]
"""

import os
import json
from typing import List, Dict, Any, Optional

from ..base_dataset import BaseRouterDataset
from ..config import TrainableRouterConfig


class DPOPreferenceDataset(BaseRouterDataset):
    """
    DPO偏好对数据集
    
    直接加载DPO偏好对格式的JSON文件
    """

    def __init__(
        self,
        config: TrainableRouterConfig,
        tokenizer=None
    ):
        """
        初始化

        Args:
            config: 数据集配置
            tokenizer: 分词器（可选，DPO训练时由Trainer处理分词）
        """
        super().__init__(config, tokenizer)
        self.strategy_names = getattr(config.model, 'strategy_names', ['no_rag', 'naive_rag'])
        self.num_strategies = len(self.strategy_names)

    def load_data(self, data_path: str):
        """
        加载DPO偏好对数据

        Args:
            data_path: DPO偏好对JSON文件路径
        """
        if not os.path.exists(data_path):
            raise ValueError(f"数据文件不存在: {data_path}")

        print(f"加载DPO偏好对数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # DPO数据格式：顶层是列表
        if isinstance(data, list):
            self.data = data
        elif isinstance(data, dict) and 'samples' in data:
            self.data = data['samples']
        else:
            raise ValueError(f"不支持的数据格式，期望列表或包含'samples'的字典")

        print(f"成功加载 {len(self.data)} 条DPO偏好对")

        # 验证数据格式并打印统计
        self._validate_and_print_stats()

    def _validate_and_print_stats(self):
        """验证数据格式并打印统计信息"""
        if len(self.data) == 0:
            print("警告: 数据集为空")
            return

        # 检查必需字段
        required_fields = ['prompt', 'chosen', 'rejected']
        sample = self.data[0]
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            raise ValueError(f"数据缺少必需字段: {missing_fields}")

        # 统计chosen分布
        chosen_counts = {}
        for item in self.data:
            chosen = item.get('chosen', 0)
            chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1

        print("\nDPO数据集统计:")
        print(f"  总样本数: {len(self.data)}")
        print("  Chosen分布（偏好策略）:")
        for idx, name in enumerate(self.strategy_names):
            count = chosen_counts.get(idx, 0)
            percentage = count / len(self.data) * 100
            print(f"    [{idx}] {name}: {count} ({percentage:.1f}%)")

        # 检查strategy_names一致性
        if 'strategy_names' in sample:
            file_strategy_names = sample['strategy_names']
            if file_strategy_names != self.strategy_names:
                print(f"警告: 配置文件中的策略名称 {self.strategy_names} "
                      f"与数据文件中的 {file_strategy_names} 不一致")
                print(f"将使用数据文件中的策略名称: {file_strategy_names}")
                self.strategy_names = file_strategy_names
                self.num_strategies = len(self.strategy_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项

        Args:
            idx: 索引

        Returns:
            DPO数据项字典，包含:
            - prompt: 问题文本
            - chosen: 偏好策略索引
            - rejected: 不喜欢策略索引
            - question_id: 问题ID
        """
        item = self.data[idx]

        return {
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected'],
            'question_id': item.get('question_id', f'q_{idx}'),
            'optimal_strategy': item.get('optimal_strategy', ''),
            'strategy_names': item.get('strategy_names', self.strategy_names),
        }

    def __len__(self) -> int:
        """数据集大小"""
        return len(self.data)

    def get_strategy_names(self) -> List[str]:
        """获取策略名称列表"""
        return self.strategy_names

    def get_num_strategies(self) -> int:
        """获取策略数量"""
        return self.num_strategies


class DPOBinaryPreferenceDataset(DPOPreferenceDataset):
    """
    二分类DPO偏好对数据集（简化版）
    
    专为二分类场景优化，假设策略为 [no_rag, naive_rag]
    """

    def __init__(
        self,
        config: TrainableRouterConfig,
        tokenizer=None
    ):
        """
        初始化
        
        Args:
            config: 数据集配置
            tokenizer: 分词器
        """
        # 强制设置为二分类
        if not hasattr(config.model, 'strategy_names'):
            config.model.strategy_names = ['no_rag', 'naive_rag']
        
        super().__init__(config, tokenizer)
        
        # 确保是二分类
        if self.num_strategies != 2:
            print(f"警告: DPOBinaryPreferenceDataset期望2个策略，当前有{self.num_strategies}个")
            print(f"将只使用前两个策略: {self.strategy_names[:2]}")
            self.strategy_names = self.strategy_names[:2]
            self.num_strategies = 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取数据项
        
        二分类简化：chosen和rejected必须是0或1
        """
        item = super().__getitem__(idx)
        
        # 验证二分类约束
        chosen = item['chosen']
        rejected = item['rejected']
        
        if chosen not in [0, 1] or rejected not in [0, 1]:
            raise ValueError(
                f"二分类DPO数据错误: chosen={chosen}, rejected={rejected}, "
                f"期望值为0或1"
            )
        
        if chosen == rejected:
            raise ValueError(
                f"DPO数据错误: chosen和rejected不能相同 ({chosen})"
            )
        
        return item
