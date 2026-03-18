"""数据集模块"""

from .base_dataset import BaseRouterDataset
from .hotpotqa_dataset import GenericRouterDataset
from .llm_judge_dataset import LLMJudgeRouterDataset
from .router_label_dataset import RouterLabelDataset
from .router_soft_label_dataset import RouterSoftLabelDataset
from .weighted_router_label_dataset import WeightedRouterLabelDataset
from .soft_label_dataset import BinarySoftLabelDataset, SoftLabelRouterDataset  # SoftLabelRouterDataset 为兼容别名
from .fusion_soft_label_dataset import FusionSoftLabelDataset, fusion_soft_label_collate_fn
from .dpo_dataset import DPOPreferenceDataset, DPOBinaryPreferenceDataset
from .decision_router_dataset import DecisionRouterDataset  # 新增：决策式路由数据集

__all__ = [
    'BaseRouterDataset',  # 抽象基类
    'GenericRouterDataset',
    'LLMJudgeRouterDataset',
    'RouterLabelDataset',
    'RouterSoftLabelDataset',  # 新增：软标签数据集
    'WeightedRouterLabelDataset',
    'BinarySoftLabelDataset',
    'SoftLabelRouterDataset',  # 兼容旧名称
    'FusionSoftLabelDataset',
    'fusion_soft_label_collate_fn',
    'DPOPreferenceDataset',
    'DPOBinaryPreferenceDataset',
    'DecisionRouterDataset',  # 新增：决策式路由数据集
]
