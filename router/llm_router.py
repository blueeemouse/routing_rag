"""
LLM Router - 基于大语言模型的路由器

支持 zero-shot 和 few-shot 模式，通过 LLM 判断查询应该使用哪种检索策略。
"""

import json
import random
import os
from typing import List, Dict, Any, Optional
import requests

from interfaces.router_interface import RouterInterface


class LLMRouter(RouterInterface):
    """
    基于大语言模型的路由器
    
    支持:
    - zero-shot: 仅使用prompt模板，无样例
    - few-shot: 使用ICL样例辅助决策
    
    路由策略:
    - no_rag: 直接回答，无需检索
    - naive_rag: 使用向量检索
    - graph_rag: 使用图检索（未来扩展）
    """
    
    # 默认的zero-shot prompt模板
    DEFAULT_ZERO_SHOT_PROMPT = """任务：确定查询处理策略

可选策略：
- no_rag: 直接回答，无需检索（适用于简单事实性问题、常识性问题、推理问题）
- naive_rag: 使用向量检索（适用于需要外部知识的问题）

规则：
1. 如果问题可以通过常识或模型内部知识回答，选择 no_rag
2. 如果问题需要查找特定信息（如具体数据、人名、地点、时间等），选择 naive_rag
3. 只输出策略名称，不要任何解释

查询：{sub_query}
策略："""

    # 默认的few-shot prompt模板
    DEFAULT_FEW_SHOT_PROMPT = """任务：确定查询处理策略

可选策略：
- no_rag: 直接回答，无需检索（适用于简单事实性问题、常识性问题、推理问题）
- naive_rag: 使用向量检索（适用于需要外部知识的问题）

规则：
1. 如果问题可以通过常识或模型内部知识回答，选择 no_rag
2. 如果问题需要查找特定信息（如具体数据、人名、地点、时间等），选择 naive_rag
3. 只输出策略名称，不要任何解释

以下是几个示例：

{examples}

查询：{sub_query}
策略："""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 20,
        mode: str = "zero_shot",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        few_shot_k: int = 5,
        examples_file: Optional[str] = None,
        prompt_template: Optional[str] = None,
        strategy_names: Optional[List[str]] = None,
    ):
        """
        初始化 LLM Router
        
        Args:
            api_url: LLM API地址
            api_key: API密钥
            model: 模型名称
            temperature: 生成温度
            max_tokens: 最大生成token数
            mode: 路由模式，"zero_shot" 或 "few_shot"
            few_shot_examples: few-shot样例列表，格式为 [{"question": "...", "strategy": "..."}]
            few_shot_k: few-shot模式下使用的样例数量
            examples_file: 样例数据文件路径（如 router_test_labels.json）
            prompt_template: 自定义prompt模板，需包含 {sub_query} 占位符
                             few-shot模式下还需包含 {examples} 占位符
            strategy_names: 可用策略名称列表
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode
        self.few_shot_k = few_shot_k
        self.strategy_names = strategy_names or ["no_rag", "naive_rag", "graph_rag"]
        
        # 处理API URL
        self._normalize_api_url()
        
        # 设置prompt模板
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = (
                self.DEFAULT_FEW_SHOT_PROMPT if mode == "few_shot" 
                else self.DEFAULT_ZERO_SHOT_PROMPT
            )
        
        # 加载或设置few-shot样例
        # 格式: {strategy_name: [...]}，按类别分组，从 strategy_names 动态获取
        self.few_shot_examples_by_strategy = {s: [] for s in self.strategy_names}
        if mode == "few_shot":
            if few_shot_examples:
                # 如果直接传入样例列表，按类别分组
                for ex in few_shot_examples:
                    strategy = ex.get("strategy")
                    if strategy in self.few_shot_examples_by_strategy:
                        self.few_shot_examples_by_strategy[strategy].append(ex)
            elif examples_file:
                self.few_shot_examples_by_strategy = self._load_examples_from_file(examples_file)
        
        # 验证模式
        if mode not in ["zero_shot", "few_shot"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'zero_shot' or 'few_shot'")
    
    def _normalize_api_url(self):
        """标准化API URL"""
        if not self.api_url.endswith("/chat/completions"):
            if self.api_url.endswith("/v1"):
                self.api_url = self.api_url + "/chat/completions"
            else:
                self.api_url = self.api_url.rstrip('/') + "/v1/chat/completions"
    
    def _load_examples_from_file(self, file_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        从文件加载样例（按类别分组）
        
        Args:
            file_path: 数据文件路径（router_test_labels.json 格式）
            
        Returns:
            按类别分组的样例字典，如 {"no_rag": [...], "naive_rag": [...]}
        """
        # 从 strategy_names 动态获取类别
        examples_by_strategy = {s: [] for s in self.strategy_names}
        
        if not os.path.exists(file_path):
            print(f"Warning: Examples file not found: {file_path}")
            return examples_by_strategy
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持 router_test_labels.json 格式
        if "samples" in data:
            for sample in data["samples"]:
                strategy = sample.get("optimal_strategy")
                # 只选择有效的策略
                if strategy in examples_by_strategy:
                    examples_by_strategy[strategy].append({
                        "question": sample["question"],
                        "strategy": strategy
                    })
        
        total = sum(len(v) for v in examples_by_strategy.values())
        print(f"Loaded {total} examples from {file_path}")
        for strategy, items in examples_by_strategy.items():
            if items:
                print(f"  - {strategy}: {len(items)} samples")
        
        return examples_by_strategy
    
    def _sample_examples(self, k: int) -> List[Dict[str, str]]:
        """
        分层采样few-shot样例（确保每个类别都有样本）
        
        Args:
            k: 总采样数量
            
        Returns:
            采样后的样例列表
        """
        sampled = []
        
        # 获取有效类别
        available_strategies = [
            s for s, items in self.few_shot_examples_by_strategy.items() 
            if len(items) > 0
        ]
        
        if not available_strategies:
            return []
        
        # 计算每个类别应采样的数量（尽量均匀分配）
        per_strategy = max(1, k // len(available_strategies))
        remaining = k
        
        for strategy in available_strategies:
            items = self.few_shot_examples_by_strategy[strategy]
            # 每个类别最多采样 per_strategy 个，但不能超过可用数量
            n = min(per_strategy, len(items), remaining)
            if n > 0:
                sampled.extend(random.sample(items, n))
                remaining -= n
        
        # 如果还有余量，从所有样例中随机补充
        if remaining > 0:
            all_items = []
            for items in self.few_shot_examples_by_strategy.values():
                all_items.extend(items)
            # 排除已采样的
            sampled_questions = {ex['question'] for ex in sampled}
            remaining_items = [ex for ex in all_items if ex['question'] not in sampled_questions]
            if remaining_items:
                n = min(remaining, len(remaining_items))
                sampled.extend(random.sample(remaining_items, n))
        
        # 打乱顺序，避免类别聚集
        random.shuffle(sampled)
        
        return sampled
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """
        格式化样例为prompt字符串
        
        Args:
            examples: 样例列表
            
        Returns:
            格式化后的字符串
        """
        formatted = []
        for ex in examples:
            formatted.append(f"查询：{ex['question']}\n策略：{ex['strategy']}")
        return "\n\n".join(formatted)
    
    def _build_prompt(self, sub_query: str) -> str:
        """
        构建完整的prompt
        
        Args:
            sub_query: 子查询
            
        Returns:
            完整的prompt字符串
        """
        if self.mode == "zero_shot":
            return self.prompt_template.format(sub_query=sub_query)
        else:  # few_shot
            sampled = self._sample_examples(self.few_shot_k)
            examples_str = self._format_examples(sampled)
            return self.prompt_template.format(
                sub_query=sub_query,
                examples=examples_str
            )
    
    def _parse_response(self, content: str) -> str:
        """
        解析LLM响应，提取策略名称
        
        Args:
            content: LLM返回的内容
            
        Returns:
            策略名称
        """
        content = content.strip().lower()
        
        # 尝试匹配策略名称
        if 'no_rag' in content or 'no rag' in content:
            return 'no_rag'
        elif 'naive_rag' in content or 'naive rag' in content:
            return 'naive_rag'
        elif 'graph_rag' in content or 'graph rag' in content:
            return 'graph_rag'
        else:
            # 默认返回 no_rag
            return 'no_rag'
    
    def route(self, sub_query: str) -> str:
        """
        路由决策：根据查询选择最佳策略
        
        Args:
            sub_query: 子查询字符串
            
        Returns:
            策略名称: 'no_rag' | 'naive_rag' | 'graph_rag'
        """
        prompt = self._build_prompt(sub_query)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', "")
            
            return self._parse_response(content)
            
        except requests.exceptions.RequestException as e:
            print(f"Error routing query: {e}")
            # 出错时默认返回 no_rag
            return 'no_rag'
    
    def route_batch(self, queries: List[str]) -> List[str]:
        """
        批量路由决策
        
        Args:
            queries: 查询列表
            
        Returns:
            策略名称列表
        """
        return [self.route(q) for q in queries]
    
    def evaluate(
        self, 
        test_data: List[Dict[str, Any]], 
        question_key: str = "question",
        label_key: str = "optimal_strategy"
    ) -> Dict[str, Any]:
        """
        评估路由器性能
        
        Args:
            test_data: 测试数据列表
            question_key: 问题字段的键名
            label_key: 标签字段的键名
            
        Returns:
            评估结果字典
        """
        correct = 0
        total = len(test_data)
        predictions = []
        
        for item in test_data:
            query = item[question_key]
            true_label = item[label_key]
            pred_label = self.route(query)
            
            predictions.append({
                "question": query,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": pred_label == true_label
            })
            
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions
        }
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'LLMRouter':
        """
        从配置字典创建实例
        
        Args:
            config_dict: 配置字典
            
        Returns:
            LLMRouter实例
        """
        return cls(
            api_url=config_dict.get("api_url", ""),
            api_key=config_dict.get("api_key", ""),
            model=config_dict.get("model", "gpt-3.5-turbo"),
            temperature=config_dict.get("temperature", 0.0),
            max_tokens=config_dict.get("max_tokens", 20),
            mode=config_dict.get("mode", "zero_shot"),
            few_shot_k=config_dict.get("few_shot_k", 5),
            examples_file=config_dict.get("examples_file"),
            prompt_template=config_dict.get("prompt_template"),
            strategy_names=config_dict.get("strategy_names"),
        )
