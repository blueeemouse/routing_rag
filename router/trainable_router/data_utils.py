"""
数据预处理工具和适配器

提供统一的数据加载和分数计算功能
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum


class StrategyMatcher:
    """策略名称匹配器"""
    
    # 策略名称到关键词的映射
    STRATEGY_PATTERNS = {
        'no_rag': ['no_rag', 'no-rag', 'norag', 'no rag', 'direct'],
        'naive_rag': ['naive_rag', 'naive-rag', 'naiverag', 'naive rag', 'vector', 'embedding'],
        'graph_rag': ['graph_rag', 'graph-rag', 'graphrag', 'graph rag', 'knowledge_graph', 'kg'],
    }
    
    @classmethod
    def match(cls, model_name: str) -> Optional[str]:
        """
        匹配模型名称到策略名称
        
        Args:
            model_name: 模型名称
            
        Returns:
            策略名称或None
        """
        model_name_lower = model_name.lower().replace('-', '_').replace(' ', '_')
        
        for strategy, patterns in cls.STRATEGY_PATTERNS.items():
            for pattern in patterns:
                pattern_lower = pattern.lower().replace('-', '_').replace(' ', '_')
                if pattern_lower in model_name_lower:
                    return strategy
        
        return None
    
    @classmethod
    def get_default_strategy_names(cls) -> List[str]:
        """获取默认策略名称列表"""
        return list(cls.STRATEGY_PATTERNS.keys())


@dataclass
class TrainingItem:
    """训练数据项（内部统一格式）"""
    question: str
    strategy_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cluster_id: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'strategy_scores': self.strategy_scores,
            'cluster_id': self.cluster_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingItem':
        return cls(
            question=data.get('question', ''),
            strategy_scores=data.get('strategy_scores', {}),
            cluster_id=data.get('cluster_id', -1),
        )


class ScoreComputer:
    """
    分数计算器
    
    支持通过公式计算分数，如：
    - 单指标: "em", "f1", "llm_judge", "bert_score"
    - 组合指标: "em * 0.3 + f1 * 0.7"
    - 性能成本权衡: "em - 0.01 * total_time"
    - 任意指标组合: "llm_judge * 0.6 + bert_score * 0.4"
    
    注意：自动支持数据中的任何指标，无需预先定义
    """
    
    def __init__(self, formula: str = "em"):
        """
        初始化分数计算器
        
        Args:
            formula: 计算公式
                - 单指标: "em", "f1", "llm_judge"
                - 组合指标: "em * 0.3 + f1 * 0.7"
                - 成本权衡: "em - 0.01 * total_time"
        """
        self.formula = formula
        self.compute_func = self._parse_formula(formula)
    
    def _parse_formula(self, formula: str) -> Callable[[Dict[str, float]], float]:
        """
        解析自定义公式
        
        动态支持数据中的任何指标。优化：只替换公式中出现的指标，避免遍历所有指标。
        
        Args:
            formula: 公式字符串，如 "em * 0.3 + f1 * 0.7"
            
        Returns:
            计算函数
        """
        # 提取公式中的所有变量名（指标名）
        # 假设指标名是字母、数字、下划线开头，后面跟着字母、数字、下划线
        formula_metrics = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula))
        
        def compute(metrics: Dict[str, float]) -> float:
            # 将指标名替换为实际值
            expr = formula
            for metric in formula_metrics:
                if metric in metrics:
                    expr = re.sub(rf'\b{metric}\b', str(metrics[metric]), expr)
            
            # 安全计算
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return float(result)
            except Exception:
                # 回退到默认：EM 和 F1 的平均
                return metrics.get('em', 0.0) * 0.5 + metrics.get('f1', 0.0) * 0.5
        
        return compute
    
    def compute(self, metrics: Dict[str, float]) -> float:
        """
        计算分数
        
        Args:
            metrics: 原始指标字典
            
        Returns:
            计算后的分数
        """
        return self.compute_func(metrics)
    
    def compute_all(self, strategy_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        计算所有策略的分数
        
        Args:
            strategy_scores: 各策略的原始指标
            
        Returns:
            各策略的计算后分数
        """
        return {
            strategy: self.compute(metrics)
            for strategy, metrics in strategy_scores.items()
        }


class DataAdapter:
    """
    数据适配器：统一不同格式
    
    支持的数据格式：
    - 单策略结果文件（现有格式）
    - 聚合格式（comparison_results）
    - LLM评判格式（未来格式）
    """
    
    def __init__(self, score_formula: str = "em"):
        """
        初始化适配器
        
        Args:
            score_formula: 分数计算公式
        """
        self.score_computer = ScoreComputer(score_formula)
    
    def _extract_predictions(self, data: Any) -> List[Dict[str, Any]]:
        """从各种数据格式中提取predictions列表"""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 尝试多种路径
            for key in ['predictions', 'results', 'data', 'items']:
                if key in data:
                    return self._extract_predictions(data[key])
            return []
        return []
    
    def _match_strategy(self, model_name: str) -> Optional[str]:
        """匹配策略名称"""
        return StrategyMatcher.match(model_name)
    
    def from_single_strategy(
        self, 
        file_path: str, 
        strategy_name: Optional[str] = None
    ) -> List[TrainingItem]:
        """
        从单个策略结果文件加载
        
        Args:
            file_path: 文件路径
            strategy_name: 策略名称（可选，自动检测）
            
        Returns:
            TrainingItem列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        predictions = self._extract_predictions(data)
        
        # 自动检测策略名称
        detected_strategy = strategy_name
        if detected_strategy is None:
            # 从model_name字段检测
            model_name = data.get('model_name', data.get('model', ''))
            detected_strategy = self._match_strategy(model_name)
        
        for pred in predictions:
            question = pred.get('question', pred.get('query', ''))
            if not question:
                continue
            
            metrics = {
                'em': pred.get('em', 0.0),
                'f1': pred.get('f1', 0.0),
                'precision': pred.get('precision', 0.0),
                'recall': pred.get('recall', 0.0),
                'total_time': pred.get('total_time', 0.0),
                'retrieval_time': pred.get('retrieval_time', 0.0),
                'generation_time': pred.get('generation_time', 0.0),
            }
            
            # 检查是否有llm_judge等额外指标
            for key in pred:
                if key.startswith('llm_judge') or key.startswith('bert_'):
                    metrics[key] = pred[key]
            
            strategy = detected_strategy or 'unknown'
            
            items.append(TrainingItem(
                question=question,
                strategy_scores={strategy: metrics}
            ))
        
        return items
    
    def from_comparison(self, file_path: str) -> List[TrainingItem]:
        """
        从comparison_results文件加载
        
        Args:
            file_path: 文件路径
            
        Returns:
            TrainingItem列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        predictions = self._extract_predictions(data)
        
        for pred in predictions:
            question = pred.get('question', '')
            if not question:
                continue
            
            # comparison格式通常没有策略区分，返回空strategy_scores
            items.append(TrainingItem(
                question=question,
                strategy_scores={}
            ))
        
        return items
    
    def from_directory(
        self, 
        dir_path: str, 
        strategy_names: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None
    ) -> List[TrainingItem]:
        """
        从目录批量加载多个策略文件
        
        Args:
            dir_path: 目录路径
            strategy_names: 策略名称列表（用于排序和匹配）
            file_patterns: 文件名模式列表
            
        Returns:
            TrainingItem列表
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在: {dir_path}")
        
        all_items = []
        
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            
            # 跳过非JSON文件
            if not filename.endswith('.json'):
                continue
            
            # 跳过comparison文件
            if 'comparison' in filename:
                continue
            
            # 匹配文件
            if file_patterns:
                if not any(pattern in filename for pattern in file_patterns):
                    continue
            
            # 尝试识别策略名称
            strategy = self._match_strategy(filename)
            
            if strategy:
                items = self.from_single_strategy(file_path, strategy)
                all_items.extend(items)
        
        return all_items
    
    def aggregate(self, items: List[TrainingItem]) -> List[TrainingItem]:
        """
        聚合多个TrainingItem（按question聚合不同策略的结果）
        
        Args:
            items: TrainingItem列表
            
        Returns:
            聚合后的TrainingItem列表
        """
        # 按question分组
        question_map: Dict[str, TrainingItem] = {}
        
        for item in items:
            question = item.question
            
            if question not in question_map:
                question_map[question] = TrainingItem(
                    question=question,
                    strategy_scores={}
                )
            
            # 合并strategy_scores
            for strategy, metrics in item.strategy_scores.items():
                if strategy not in question_map[question].strategy_scores:
                    question_map[question].strategy_scores[strategy] = {}
                question_map[question].strategy_scores[strategy].update(metrics)
        
        return list(question_map.values())
    
    def filter_by_coverage(
        self, 
        items: List[TrainingItem], 
        min_strategies: int = 2
    ) -> List[TrainingItem]:
        """
        过滤掉策略覆盖不足的样本
        
        Args:
            items: TrainingItem列表
            min_strategies: 最少策略数量
            
        Returns:
            过滤后的列表
        """
        return [
            item for item in items 
            if len(item.strategy_scores) >= min_strategies
        ]
    
    def normalize_scores(
        self, 
        items: List[TrainingItem], 
        strategy: Optional[str] = None
    ) -> List[TrainingItem]:
        """
        归一化分数
        
        Args:
            items: TrainingItem列表
            strategy: 指定策略（None表示所有策略）
            
        Returns:
            归一化后的列表
        """
        # 收集所有分数
        all_scores = []
        
        for item in items:
            strategies = item.strategy_scores.keys() if strategy is None else [strategy]
            
            for strat in strategies:
                if strat in item.strategy_scores:
                    score = self.score_computer.compute(item.strategy_scores[strat])
                    all_scores.append((item.question, strat, score))
        
        if not all_scores:
            return items
        
        # 计算全局最大值
        max_score = max(s for _, _, s in all_scores)
        min_score = min(s for _, _, s in all_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        # 归一化
        for item in items:
            strategies = item.strategy_scores.keys() if strategy is None else [strategy]
            
            for strat in strategies:
                if strat in item.strategy_scores:
                    original_score = self.score_computer.compute(item.strategy_scores[strat])
                    normalized = (original_score - min_score) / score_range
                    
                    # 更新指标
                    item.strategy_scores[strat]['_raw_score'] = original_score
                    item.strategy_scores[strat]['_normalized_score'] = normalized
        
        return items


def aggregate_by_query(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    按query聚合数据（旧API，保留兼容性）
    
    Args:
        data: 原始数据列表
        
    Returns:
        {query: {model_name: performance}}
    """
    aggregated = {}
    
    for item in data:
        query = item.get('query', item.get('question', ''))
        if not query:
            continue
        
        model_name = item.get('model_name', item.get('strategy', ''))
        performance = item.get('performance', item.get('em', 0.0))
        
        if query not in aggregated:
            aggregated[query] = {}
        
        aggregated[query][model_name] = performance
    
    return aggregated


def save_training_data(
    data: List[TrainingItem], 
    output_path: str,
    format: str = 'jsonl'
):
    """
    保存训练数据
    
    Args:
        data: TrainingItem列表
        output_path: 输出路径
        format: 保存格式 ('json' or 'jsonl')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([item.to_dict() for item in data], f, ensure_ascii=False, indent=2)


def load_training_data(input_path: str) -> List[TrainingItem]:
    """
    加载训练数据
    
    Args:
        input_path: 输入路径
        
    Returns:
        TrainingItem列表
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.endswith('.jsonl'):
            data = [TrainingItem.from_dict(json.loads(line)) for line in f]
        else:
            data = [TrainingItem.from_dict(item) for item in json.load(f)]
    
    return data