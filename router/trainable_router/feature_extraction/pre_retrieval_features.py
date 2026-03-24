"""
Pre-Retrieval 特征提取器

从检索结果中提取特征，用于辅助路由决策。

特征包括：
1. 相似度统计特征（5 维）
2. Top-k 分数（3 维）
3. 差距指标（2 维）

总计：10 维
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np


class PreRetrievalFeatureExtractor:
    """
    Pre-Retrieval 特征提取器
    
    从检索结果中提取 10 维特征：
    - 相似度统计（5 维）：max, min, avg, std, range
    - Top-k 分数（3 维）：top1, top2, top3
    - 差距指标（2 维）：top1_top2_gap, top1_threshold_gap
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        初始化特征提取器
        
        Args:
            similarity_threshold: 相似度阈值，用于计算 top1_threshold_gap
        """
        self.similarity_threshold = similarity_threshold
        
        # 特征名称列表（按顺序）
        self.feature_names = [
            # 相似度统计特征（5 维）
            'max_similarity',
            'min_similarity',
            'avg_similarity',
            'similarity_std',
            'similarity_range',
            
            # Top-k 分数（3 维）
            'top1_score',
            'top2_score',
            'top3_score',
            
            # 差距指标（2 维）
            'top1_top2_gap',
            'top1_threshold_gap',
        ]
        
        self.num_features = len(self.feature_names)
    
    def extract_from_nodes(self, nodes: List[Any]) -> Dict[str, float]:
        """
        从检索结果节点中提取特征
        
        Args:
            nodes: LlamaIndex 检索返回的节点列表，每个节点应有 score 属性
        
        Returns:
            特征字典
        """
        # 提取相似度分数
        scores = []
        for node in nodes:
            if hasattr(node, 'score') and node.score is not None:
                scores.append(float(node.score))
            elif hasattr(node, 'get_score') and node.get_score() is not None:
                scores.append(float(node.get_score()))
        
        # 如果没有分数，返回全零特征
        if not scores:
            return self._zero_features()
        
        # 确保至少有 3 个分数（不足则填充）
        while len(scores) < 3:
            scores.append(0.0)
        
        # 排序（从高到低）
        scores = sorted(scores, reverse=True)[:3]
        
        # 计算特征
        features = {}
        
        # 1. 相似度统计特征
        scores_array = np.array(scores)
        features['max_similarity'] = float(np.max(scores_array))
        features['min_similarity'] = float(np.min(scores_array))
        features['avg_similarity'] = float(np.mean(scores_array))
        features['similarity_std'] = float(np.std(scores_array)) if len(scores) > 1 else 0.0
        features['similarity_range'] = features['max_similarity'] - features['min_similarity']
        
        # 2. Top-k 分数
        features['top1_score'] = scores[0] if len(scores) > 0 else 0.0
        features['top2_score'] = scores[1] if len(scores) > 1 else 0.0
        features['top3_score'] = scores[2] if len(scores) > 2 else 0.0
        
        # 3. 差距指标
        features['top1_top2_gap'] = scores[0] - scores[1] if len(scores) > 1 else 0.0
        features['top1_threshold_gap'] = scores[0] - self.similarity_threshold
        
        return features
    
    def extract_from_scores(self, scores: List[float]) -> Dict[str, float]:
        """
        直接从相似度分数列表中提取特征
        
        Args:
            scores: 相似度分数列表
        
        Returns:
            特征字典
        """
        # 确保至少有 3 个分数（不足则填充）
        scores = list(scores)
        while len(scores) < 3:
            scores.append(0.0)
        
        # 排序（从高到低）
        scores = sorted(scores, reverse=True)[:3]
        
        # 计算特征
        features = {}
        
        # 1. 相似度统计特征
        scores_array = np.array(scores)
        features['max_similarity'] = float(np.max(scores_array))
        features['min_similarity'] = float(np.min(scores_array))
        features['avg_similarity'] = float(np.mean(scores_array))
        features['similarity_std'] = float(np.std(scores_array)) if len(scores) > 1 else 0.0
        features['similarity_range'] = features['max_similarity'] - features['min_similarity']
        
        # 2. Top-k 分数
        features['top1_score'] = scores[0] if len(scores) > 0 else 0.0
        features['top2_score'] = scores[1] if len(scores) > 1 else 0.0
        features['top3_score'] = scores[2] if len(scores) > 2 else 0.0
        
        # 3. 差距指标
        features['top1_top2_gap'] = scores[0] - scores[1] if len(scores) > 1 else 0.0
        features['top1_threshold_gap'] = scores[0] - self.similarity_threshold
        
        return features
    
    def _zero_features(self) -> Dict[str, float]:
        """返回全零特征（当没有检索结果时使用）"""
        return {name: 0.0 for name in self.feature_names}
    
    def extract_batch(self, all_nodes: List[List[Any]]) -> torch.Tensor:
        """
        批量提取特征
        
        Args:
            all_nodes: 检索结果列表的列表，外层是 batch，内层是每个 query 的检索结果
        
        Returns:
            特征张量 (batch_size, num_features)
        """
        features_list = []
        
        for nodes in all_nodes:
            feat_dict = self.extract_from_nodes(nodes)
            # 按照 feature_names 的顺序提取特征值
            feat_vec = [feat_dict.get(name, 0.0) for name in self.feature_names]
            features_list.append(feat_vec)
        
        features_array = np.array(features_list, dtype=np.float32)
        return torch.tensor(features_array, dtype=torch.float32)
    
    def extract_batch_from_scores(self, all_scores: List[List[float]]) -> torch.Tensor:
        """
        批量从分数列表中提取特征
        
        Args:
            all_scores: 相似度分数列表的列表，外层是 batch，内层是每个 query 的分数
        
        Returns:
            特征张量 (batch_size, num_features)
        """
        features_list = []
        
        for scores in all_scores:
            feat_dict = self.extract_from_scores(scores)
            # 按照 feature_names 的顺序提取特征值
            feat_vec = [feat_dict.get(name, 0.0) for name in self.feature_names]
            features_list.append(feat_vec)
        
        features_array = np.array(features_list, dtype=np.float32)
        return torch.tensor(features_array, dtype=torch.float32)


class PreRetrievalFeatureNormalizer:
    """
    Pre-Retrieval 特征归一化器
    
    使用 Z-score 标准化：(x - mean) / std
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        初始化归一化器
        
        Args:
            feature_names: 特征名称列表（用于保存统计信息时的键名）
        """
        self.feature_names = feature_names or [
            'max_similarity', 'min_similarity', 'avg_similarity',
            'similarity_std', 'similarity_range',
            'top1_score', 'top2_score', 'top3_score',
            'top1_top2_gap', 'top1_threshold_gap',
        ]
        
        # 特征统计信息（均值和标准差）
        self.feature_stats: Dict[str, Tuple[float, float]] = {}
    
    def fit(self, features: torch.Tensor):
        """
        计算特征的统计信息（均值和标准差）
        
        Args:
            features: 特征张量 (num_samples, num_features)
        """
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        
        for i, name in enumerate(self.feature_names):
            mean = float(np.mean(features_np[:, i]))
            std = float(np.std(features_np[:, i]))
            # 避免 std 为 0
            if std < 1e-8:
                std = 1.0
            self.feature_stats[name] = (mean, std)
        
        return self
    
    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        归一化特征
        
        Args:
            features: 特征张量 (batch_size, num_features)
        
        Returns:
            归一化后的特征张量
        """
        if not self.feature_stats:
            # 如果没有拟合过，直接返回原特征
            return features
        
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        features_normalized = features_np.copy()
        
        for i, name in enumerate(self.feature_names):
            if name in self.feature_stats:
                mean, std = self.feature_stats[name]
                if std > 1e-8:
                    features_normalized[:, i] = (features_np[:, i] - mean) / std
        
        return torch.tensor(features_normalized, dtype=torch.float32)
    
    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        先拟合再归一化
        
        Args:
            features: 特征张量 (num_samples, num_features)
        
        Returns:
            归一化后的特征张量
        """
        self.fit(features)
        return self.transform(features)
    
    def save_stats(self, path: str):
        """
        保存特征统计信息到文件
        
        Args:
            path: 保存路径
        """
        import json
        # 转换为可序列化的格式
        stats_serializable = {
            name: {'mean': mean, 'std': std}
            for name, (mean, std) in self.feature_stats.items()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
    
    def load_stats(self, path: str):
        """
        从文件加载特征统计信息
        
        Args:
            path: 文件路径
        """
        import json
        with open(path, 'r', encoding='utf-8') as f:
            stats_serializable = json.load(f)
        
        self.feature_stats = {
            name: (item['mean'], item['std'])
            for name, item in stats_serializable.items()
        }


# 便捷函数
def extract_pre_retrieval_features(nodes: List[Any], threshold: float = 0.5) -> Dict[str, float]:
    """
    便捷函数：从检索结果中提取特征
    
    Args:
        nodes: 检索结果节点列表
        threshold: 相似度阈值
    
    Returns:
        特征字典
    """
    extractor = PreRetrievalFeatureExtractor(similarity_threshold=threshold)
    return extractor.extract_from_nodes(nodes)


def extract_pre_retrieval_features_from_scores(
    scores: List[float], 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    便捷函数：从分数列表中提取特征
    
    Args:
        scores: 相似度分数列表
        threshold: 相似度阈值
    
    Returns:
        特征字典
    """
    extractor = PreRetrievalFeatureExtractor(similarity_threshold=threshold)
    return extractor.extract_from_scores(scores)
