"""路由器推理模块"""

from .dc_router import DCRouter
from .dpo_router import DPORouter
from .internal_representation_router import InternalRepresentationRouter

__all__ = ['DCRouter', 'DPORouter', 'InternalRepresentationRouter']
