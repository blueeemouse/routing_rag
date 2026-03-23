"""
InternalRepresentationRouter 推理类

用于端到端测试的内部表征路由器实现。
封装了 LLM 表征提取和路由决策，使其可以像 DCRouter 一样直接使用。

工作流程:
    Query -> LLM -> 提取内部表征 -> MLP 分类器 -> 路由决策

支持两种模式:
    1. 实时提取: 加载 LLM 实时提取表征（需要 GPU 内存）
    2. 预加载表征: 预先提取表征，直接加载使用（推荐）
"""

import os
import json
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from interfaces.router_interface import RouterInterface
from ..models.internal_representation_router_model import InternalRepresentationRouterModel


class InternalRepresentationRouter(RouterInterface):
    """内部表征路由器推理类"""
    
    def __init__(
        self,
        model_path: str,
        # 预加载表征模式参数
        representations_path: Optional[str] = None,
        questions_file: Optional[str] = None,
        # 实时提取模式参数
        llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        representation_type: str = "deep_last_token",
        layer_ids: Optional[List[int]] = None,
        device: str = "auto",
        llm_device: str = "auto",
        dtype: str = "float16",
        **kwargs
    ):
        """
        初始化
        
        Args:
            model_path: 训练好的 MLP 模型路径
            
            # 预加载表征模式（推荐）:
            representations_path: 预提取的表征目录路径（如 outputs/representations/fp16_qwen2.5-3b-instruct）
            questions_file: 问题文件路径（JSONL 格式，用于匹配表征）
            
            # 实时提取模式:
            llm_model_name: LLM 模型名称（用于提取表征）
            representation_type: 表征类型
            layer_ids: 自定义层号 [shallow, middle, deep]，默认 [11, 23, 35]
            device: MLP 模型设备
            llm_device: LLM 设备
            dtype: LLM 数据类型 (float16, bfloat16, float32)
        """
        self.model_path = model_path
        self.representations_path = representations_path
        self.questions_file = questions_file
        self.llm_model_name = llm_model_name
        self.representation_type = representation_type
        self.device = device
        self.llm_device = llm_device
        self.dtype = dtype
        
        # 默认层号 (Qwen2.5-3B 共 36 层, 0-indexed: 0-35)
        self.layer_ids = layer_ids if layer_ids else [11, 23, 35]
        
        # 加载 MLP 模型配置
        self._load_config()
        
        # 初始化 MLP 模型
        self._init_mlp_model()
        
        # 预加载表征模式
        self.preloaded_representations = None
        self.metadata = []
        self.question_to_idx = {}
        self.question_to_repr = {}  # 兼容旧格式
        self._use_preloaded = False
        
        if representations_path:
            self._load_preloaded_representations()
        else:
            # 实时提取模式：加载 LLM（延迟加载）
            self.llm = None
            self.tokenizer = None
            self._llm_loaded = False
        
    def _load_config(self):
        """加载 MLP 模型配置"""
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.mlp_config = json.load(f)
        else:
            # 默认配置
            self.mlp_config = {
                'representation_dim': 2048,
                'hidden_size': 512,
                'strategy_names': ['no_rag', 'naive_rag'],
                'num_strategies': 2,
            }
    
    def _init_mlp_model(self):
        """初始化 MLP 模型"""
        from ..config import ModelConfig
        
        # 创建模型配置（不传 model_type，它不是 ModelConfig 的参数）
        config = ModelConfig(
            hidden_size=self.mlp_config.get('hidden_size', 512),
            strategy_names=self.mlp_config.get('strategy_names', ['no_rag', 'naive_rag']),
            num_strategies=self.mlp_config.get('num_strategies', 2),
        )
        config.representation_dim = self.mlp_config.get('representation_dim', 2048)
        
        # 创建模型
        self.mlp_model = InternalRepresentationRouterModel(
            config,
            representation_dim=config.representation_dim
        )
        
        # 加载权重
        model_file = os.path.join(self.model_path, 'model.pt')
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.mlp_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.mlp_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.mlp_model.load_state_dict(checkpoint)
            print(f"已加载 MLP 模型权重: {self.model_path}")
        
        # 设置设备
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mlp_model.to(self.device)
        self.mlp_model.eval()
        
        print(f"MLP 模型配置:")
        print(f"  - 表征维度: {config.representation_dim}")
        print(f"  - 隐藏维度: {config.hidden_size}")
        print(f"  - 策略数量: {config.num_strategies}")
        print(f"  - 设备: {self.device}")
    
    def _load_preloaded_representations(self):
        """加载预提取的表征
        
        存储格式:
        - shard_XXXX.npz: 每个 shard 包含多种表征类型
          - keys: shallow_mean, shallow_last_token, middle_mean, middle_last_token,
                  deep_mean, deep_last_token, next_token_logits
        - metadata.json: 包含问题列表和标签信息
        """
        print(f"加载预提取表征: {self.representations_path}")
        
        # 查找所有 shard 文件
        import glob
        shard_files = sorted(glob.glob(os.path.join(self.representations_path, "shard_*.npz")))
        
        if not shard_files:
            raise FileNotFoundError(
                f"找不到 shard 文件: {self.representations_path}/shard_*.npz\n"
                f"请确保表征目录正确，或先运行 collect_internal_representations.py 提取表征"
            )
        
        print(f"  - 找到 {len(shard_files)} 个 shard 文件")
        
        # 加载并合并所有 shard 的指定表征类型
        all_representations = []
        for shard_file in shard_files:
            data = np.load(shard_file)
            if self.representation_type not in data:
                raise KeyError(
                    f"表征类型 '{self.representation_type}' 不存在于 {shard_file}\n"
                    f"可用的表征类型: {list(data.keys())}"
                )
            all_representations.append(data[self.representation_type])
        
        # 合并所有表征
        self.preloaded_representations = np.concatenate(all_representations, axis=0)
        print(f"  - 表征类型: {self.representation_type}")
        print(f"  - 表征形状: {self.preloaded_representations.shape}")
        
        # 加载 metadata.json 获取问题列表
        metadata_path = os.path.join(self.representations_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"  - 加载 metadata: {len(self.metadata)} 条记录")
            
            # 建立问题到索引的映射
            self.question_to_idx = {}
            for idx, item in enumerate(self.metadata):
                question = item.get('question', '')
                if question:
                    self.question_to_idx[question] = idx
        else:
            self.metadata = []
            self.question_to_idx = {}
            print(f"  - 警告: 未找到 metadata.json，将使用索引顺序")
        
        # 如果提供了 questions_file，建立问题映射
        if self.questions_file:
            self._build_question_mapping()
        
        self._use_preloaded = True
        print(f"  - 模式: 预加载表征模式")
    
    def _build_question_mapping(self):
        """建立问题到表征索引的映射"""
        print(f"加载问题文件: {self.questions_file}")
        
        import json
        
        questions = []
        if self.questions_file.endswith('.jsonl'):
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    questions.append(data.get('question', ''))
        elif self.questions_file.endswith('.json'):
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'samples' in data:
                    questions = [s.get('question', '') for s in data['samples']]
                elif isinstance(data, list):
                    questions = [s.get('question', '') if isinstance(s, dict) else s for s in data]
        
        if len(questions) != len(self.preloaded_representations):
            print(f"  警告: 问题数量 ({len(questions)}) 与表征数量 ({len(self.preloaded_representations)}) 不匹配")
        
        # 建立映射（问题文本 -> 索引）
        for i, q in enumerate(questions):
            self.question_to_repr[q] = i
        
        print(f"  - 已建立 {len(self.question_to_repr)} 个问题映射")
    
    def _load_llm(self):
        """延迟加载 LLM"""
        if self._llm_loaded:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 设置设备
        if self.llm_device == 'auto':
            self.llm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 设置数据类型
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        dtype = dtype_map.get(self.dtype, torch.float16)
        
        print(f"加载 LLM: {self.llm_model_name}")
        print(f"  - dtype: {self.dtype}, device: {self.llm_device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name, 
            trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.llm = self.llm.to(self.llm_device)
        self.llm.eval()
        
        print(f"  - LLM 加载完成. 层数: {self.llm.config.num_hidden_layers}")
        
        self._llm_loaded = True
    
    def _extract_representation(self, query: str) -> torch.Tensor:
        """
        提取单个 query 的内部表征
        
        Args:
            query: 查询文本
            
        Returns:
            表征向量 tensor, shape: (1, dim)
        """
        return self._extract_representations_batch([query])
    
    def _extract_representations_batch(self, queries: List[str]) -> torch.Tensor:
        """
        批量提取内部表征
        
        Args:
            queries: 查询文本列表
            
        Returns:
            表征向量 tensor, shape: (batch, dim)
        """
        # 确保 LLM 已加载
        self._load_llm()
        
        # 构造输入
        messages_list = [[{"role": "user", "content": q}] for q in queries]
        
        # Tokenize
        all_input_ids = []
        all_attention_masks = []
        
        for messages in messages_list:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors=None,
            )
            all_input_ids.append(input_ids)
            all_attention_masks.append([1] * len(input_ids))
        
        # 右 padding
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_masks.append(mask + [0] * pad_len)
        
        input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(padded_attention_masks, dtype=torch.long)
        
        if self.llm_device != 'cpu':
            input_ids = input_ids.to(self.llm_device)
            attention_mask = attention_mask.to(self.llm_device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        hidden_states = outputs.hidden_states
        batch_size = input_ids.shape[0]
        
        # 根据表征类型提取
        representation = self._get_representation_by_type(
            hidden_states, attention_mask, batch_size
        )
        
        return representation
    
    def _get_representation_by_type(
        self, 
        hidden_states: tuple, 
        attention_mask: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        根据类型获取表征
        
        Args:
            hidden_states: 各层隐藏状态
            attention_mask: 注意力掩码
            batch_size: 批大小
            
        Returns:
            表征向量
        """
        layer_names = {
            self.layer_ids[0]: 'shallow',
            self.layer_ids[1]: 'middle', 
            self.layer_ids[2]: 'deep'
        }
        
        # 提取各层表征
        layer_repr = {}
        for layer_id in self.layer_ids:
            name = layer_names[layer_id]
            layer_hidden = hidden_states[layer_id + 1].cpu().to(torch.float32)
            
            # Mean pooling
            mask_expanded = attention_mask.cpu().unsqueeze(-1).to(torch.float32)
            masked_sum = (layer_hidden * mask_expanded).sum(dim=1)
            mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = masked_sum / mask_sum
            
            # Last token
            last_token_indices = attention_mask.cpu().sum(dim=1) - 1
            last_token_indices = last_token_indices.clamp(min=0)
            last_token_hidden = layer_hidden[
                torch.arange(batch_size), last_token_indices
            ]
            
            layer_repr[f'{name}_mean'] = mean_pooled
            layer_repr[f'{name}_last_token'] = last_token_hidden
        
        # 根据类型组合
        repr_type = self.representation_type
        
        if repr_type in layer_repr:
            # 单层表征
            result = layer_repr[repr_type]
        elif repr_type == 'concat_deep':
            # 拼接 deep 的 mean 和 last_token
            result = torch.cat([
                layer_repr['deep_mean'],
                layer_repr['deep_last_token']
            ], dim=-1)
        elif repr_type == 'concat_all_mean':
            result = torch.cat([
                layer_repr['shallow_mean'],
                layer_repr['middle_mean'],
                layer_repr['deep_mean']
            ], dim=-1)
        elif repr_type == 'concat_all_last':
            result = torch.cat([
                layer_repr['shallow_last_token'],
                layer_repr['middle_last_token'],
                layer_repr['deep_last_token']
            ], dim=-1)
        elif repr_type == 'concat_all':
            result = torch.cat([
                layer_repr['shallow_mean'],
                layer_repr['shallow_last_token'],
                layer_repr['middle_mean'],
                layer_repr['middle_last_token'],
                layer_repr['deep_mean'],
                layer_repr['deep_last_token']
            ], dim=-1)
        else:
            # 默认使用 deep_last_token
            result = layer_repr['deep_last_token']
        
        return result
    
    def route(self, sub_query: str) -> str:
        """
        路由决策：根据 query 选择最佳策略
        
        Args:
            sub_query: 子查询字符串
            
        Returns:
            策略名称: 'no_rag' | 'naive_rag' | 'graph_rag'
        """
        routes = self.route_batch([sub_query])
        return routes[0]
    
    def route_batch(self, queries: List[str]) -> List[str]:
        """
        批量路由决策
        
        Args:
            queries: query 字符串列表
            
        Returns:
            策略名称列表
        """
        # 获取表征
        if self._use_preloaded:
            # 预加载模式：从预提取的表征中查找
            representations = self._get_preloaded_representations(queries)
        else:
            # 实时提取模式
            representations = self._extract_representations_batch(queries)
        
        representations = representations.to(self.device)
        
        # MLP 分类
        with torch.no_grad():
            logits = self.mlp_model(representations)
            predicted_indices = logits.argmax(dim=-1)
            
            strategy_names = self.mlp_config.get(
                'strategy_names', 
                ['no_rag', 'naive_rag']
            )
            routes = [strategy_names[idx.item()] for idx in predicted_indices]
        
        return routes
    
    def _get_preloaded_representations(self, queries: List[str]) -> torch.Tensor:
        """
        从预加载的表征中获取
        
        Args:
            queries: query 字符串列表
            
        Returns:
            表征向量 tensor
        """
        if self.preloaded_representations is None:
            raise ValueError("未加载预提取表征")
        
        batch_size = len(queries)
        repr_dim = self.preloaded_representations.shape[1]
        result = np.zeros((batch_size, repr_dim), dtype=np.float32)
        
        for i, query in enumerate(queries):
            # 使用 question_to_idx 映射（新格式）
            if query in self.question_to_idx:
                idx = self.question_to_idx[query]
                result[i] = self.preloaded_representations[idx]
            # 兼容旧的 question_to_repr 映射
            elif hasattr(self, 'question_to_repr') and query in self.question_to_repr:
                idx = self.question_to_repr[query]
                result[i] = self.preloaded_representations[idx]
            else:
                # 如果找不到匹配的问题，打印警告
                print(f"警告: 未找到问题的预提取表征: {query[:50]}...")
                # 使用零向量（或者可以抛出异常）
                result[i] = 0
        
        return torch.tensor(result, dtype=torch.float32)
    
    def to(self, device: str):
        """移动 MLP 模型到设备"""
        self.device = device
        self.mlp_model.to(device)
    
    @property
    def strategy_names(self) -> List[str]:
        """获取策略名称列表"""
        return self.mlp_config.get('strategy_names', ['no_rag', 'naive_rag'])
    
    @property
    def mlp_device(self) -> torch.device:
        """获取 MLP 设备"""
        return next(self.mlp_model.parameters()).device
