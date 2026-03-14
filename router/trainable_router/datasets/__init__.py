"""数据集模块"""

from .hotpotqa_dataset import GenericRouterDataset
from .llm_judge_dataset import LLMJudgeRouterDataset
from .router_label_dataset import RouterLabelDataset
from .weighted_router_label_dataset import WeightedRouterLabelDataset
from .soft_label_dataset import BinarySoftLabelDataset, SoftLabelRouterDataset  # SoftLabelRouterDataset 为兼容别名
from .fusion_soft_label_dataset import FusionSoftLabelDataset, fusion_soft_label_collate_fn
from .dpo_dataset import DPOPreferenceDataset, DPOBinaryPreferenceDataset

__all__ = [
    'GenericRouterDataset', 
    'LLMJudgeRouterDataset', 
    'RouterLabelDataset', 
    'WeightedRouterLabelDataset',
    'BinarySoftLabelDataset',
    'SoftLabelRouterDataset',  # 兼容旧名称
    'FusionSoftLabelDataset',
    'fusion_soft_label_collate_fn',
    'DPOPreferenceDataset',
    'DPOBinaryPreferenceDataset',
]
