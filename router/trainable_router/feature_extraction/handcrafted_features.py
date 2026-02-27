"""
手工特征提取器 - 基于RouteRAG论文3.2节

提取4大类特征：
1. Syntactic Structure Features（句法结构特征）- 基于选区解析树
2. Dependency-based Structural Features（依存结构特征）
3. Semantic and Lexical Features（语义词汇特征）
4. Tree Structure and Interaction Features（树结构和交互特征）

参考文献：
- RouteRAG: Learning to Route Queries for RAG, Section 3.2
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from collections import Counter


class HandcraftedFeatureExtractor:
    """
    手工特征提取器（基于RouteRAG论文）
    
    特征维度：约85维（原始特征>80维，通过特征选择得到top-k）
    
    论文特征体系：
    1. Syntactic Structure Features (9 counts + 12 ratios = 21维)
    2. Dependency-based Structural Features (约20维)
    3. Semantic and Lexical Features (约25维)
    4. Tree Structure and Interaction Features (约20维)
    """
    
    def __init__(
        self, 
        use_spacy: bool = True,
        spacy_model: str = 'en_core_web_sm',
        normalize: bool = True,
        feature_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        selected_features: Optional[List[str]] = None
    ):
        """
        初始化特征提取器
        
        Args:
            use_spacy: 是否使用spaCy进行依存分析和NER
            spacy_model: spaCy模型名称
            normalize: 是否对特征进行归一化
            feature_stats: 特征统计信息（均值和标准差），用于归一化
                          格式: {'feature_name': (mean, std), ...}
            selected_features: 选择的特征名称列表（用于特征选择）
                              如果为None，使用所有特征
        """
        self.use_spacy = use_spacy
        self.normalize = normalize
        self.feature_stats = feature_stats
        self.selected_features = selected_features
        
        # 初始化spaCy
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(spacy_model)
                print(f"✓ 已加载spaCy模型: {spacy_model}")
            except Exception as e:
                print(f"⚠ 无法加载spaCy模型 {spacy_model}: {e}")
                print("  将使用轻量级规则方法代替")
                self.use_spacy = False
                self.nlp = None
        
        # 定义特征名称（按论文4大类组织）
        self.feature_names = self._define_feature_names()
        
        # 疑问词词典
        self.question_words = {
            'what', 'who', 'which', 'where', 'when',  # 简单事实查询
            'how', 'why',  # 复杂推理查询
        }
        
        # 否定词词典
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 
            'neither', 'nor', "n't", 'without', 'hardly', 'barely'
        }
        
        # 功能词列表（用于content-to-function ratio）
        self.function_pos = {'DET', 'ADP', 'CCONJ', 'SCONJ', 'PRON', 'PART', 'INTJ'}
    
    def _define_feature_names(self) -> List[str]:
        """定义所有特征名称"""
        features = []
        
        # ========== 1. Syntactic Structure Features ==========
        # 基础计数特征（9维）
        features.extend([
            'num_words',  # W
            'num_sentences',  # S
            'num_clauses',  # C
            'num_dependent_clauses',  # DC
            'num_tunits',  # T
            'num_complex_tunits',  # CT
            'num_coordinate_phrases',  # CP
            'num_complex_nominals',  # CN
            'num_verb_phrases',  # VP
        ])
        
        # 比率特征（12维）
        features.extend([
            'mean_length_sentence',  # MLS = W/S
            'mean_length_tunit',  # MLT = W/T
            'sentence_complexity',  # C/S
            'subordination_ratio_C_T',  # C/T
            'subordination_ratio_CT_T',  # CT/T
            'subordination_ratio_DC_C',  # DC/C
            'subordination_ratio_DC_T',  # DC/T
            'coordination_ratio_CP_C',  # CP/C
            'coordination_ratio_CP_T',  # CP/T
            'tunit_per_sentence',  # T/S
            'phrasal_sophistication_CN_C',  # CN/C
            'phrasal_sophistication_CN_T',  # CN/T
            'phrasal_sophistication_VP_T',  # VP/T
        ])
        
        # ========== 2. Dependency-based Structural Features ==========
        features.extend([
            # 依存距离特征
            'max_dependency_distance',
            'avg_dependency_distance',
            'num_long_range_dependencies',  # distance > 5
            
            # 关系类型计数
            'num_subject_verb',
            'num_object_verb',
            'num_modifier',
            'num_coordination_dep',
            'num_subordination_dep',
            
            # 树不平衡度
            'tree_imbalance',
        ])
        
        # ========== 3. Semantic and Lexical Features ==========
        features.extend([
            # 命名实体特征
            'num_entities',
            'num_person_entities',
            'num_org_entities',
            'num_location_entities',
            'num_date_entities',
            'entity_density',  # entities per token
            
            # 语义角色特征（简化版）
            'num_agents',
            'num_patients',
            'num_temporal',
            'num_locative',
            
            # 词汇多样性
            'lexical_diversity',  # unique token ratio
            'content_function_ratio',
            'information_density',
            
            # 问题类型指示器
            'is_question_word',
            'is_simple_question',  # what/who/which/where/when
            'is_complex_question',  # how/why
            
            # 复杂度标记
            'has_coordination',
            'has_subordination',
            'has_negation',
            'has_passive_voice',
        ])
        
        # ========== 4. Tree Structure and Interaction Features ==========
        features.extend([
            # 树全局属性
            'max_tree_depth',
            'max_tree_width',
            'leaf_nonleaf_ratio',
            'branching_factor',
            'depth_width_ratio',
            
            # 交互特征
            'tokens_per_clause',
            'entities_per_token',
            'depth_per_token',
            'connectors_per_clause',
            
            # 额外交互特征
            'complex_nominals_per_clause',
            'verb_phrases_per_tunit',
            'modifiers_per_token',
        ])
        
        return features
    
    def extract_features(self, query: str) -> Dict[str, float]:
        """
        提取单个query的所有特征
        
        Args:
            query: 查询字符串
            
        Returns:
            特征字典 {feature_name: value}
        """
        features = {}
        
        if self.use_spacy and self.nlp is not None:
            # 使用spaCy进行深度分析
            doc = self.nlp(query)
            features.update(self._extract_syntactic_features(doc))
            features.update(self._extract_dependency_features(doc))
            features.update(self._extract_semantic_features(doc))
            features.update(self._extract_tree_features(doc))
        else:
            # 使用轻量级规则方法
            features.update(self._extract_syntactic_features_rules(query))
            features.update(self._extract_dependency_features_rules(query))
            features.update(self._extract_semantic_features_rules(query))
            features.update(self._extract_tree_features_rules(query))
        
        # 特征选择（如果指定）
        if self.selected_features is not None:
            features = {k: v for k, v in features.items() if k in self.selected_features}
        
        return features
    
    def extract_batch(self, queries: List[str]) -> torch.Tensor:
        """
        批量提取特征
        
        Args:
            queries: query字符串列表
            
        Returns:
            特征张量 (batch_size, num_features)
        """
        features_list = []
        
        for query in queries:
            feat_dict = self.extract_features(query)
            # 按照feature_names的顺序提取特征值
            feat_vec = [feat_dict.get(name, 0.0) for name in self.feature_names]
            features_list.append(feat_vec)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        # 归一化
        if self.normalize:
            features_array = self._normalize_features(features_array)
        
        return torch.tensor(features_array, dtype=torch.float32)
    
    # ========== 1. Syntactic Structure Features ==========
    
    def _extract_syntactic_features(self, doc) -> Dict[str, float]:
        """使用spaCy提取句法结构特征"""
        features = {}
        
        # 基础计数
        tokens = [t for t in doc if not t.is_punct]
        features['num_words'] = float(len(tokens))
        features['num_sentences'] = float(len(list(doc.sents)))
        
        # 从句计数（通过依存关系）
        clauses = [t for t in doc if t.dep_ in {'advcl', 'acl', 'relcl', 'csubj', 'ccomp'}]
        features['num_clauses'] = float(len(clauses))
        
        # 从属从句
        dependent_clauses = [t for t in doc if t.dep_ in {'advcl', 'ccomp'}]
        features['num_dependent_clauses'] = float(len(dependent_clauses))
        
        # T-units（主句及其从属从句）
        roots = [t for t in doc if t.head == t]
        features['num_tunits'] = float(len(roots))
        
        # 复杂T-units（包含从属从句的T-unit）
        complex_tunits = 0
        for root in roots:
            has_subordinate = any(t.dep_ in {'advcl', 'ccomp'} for t in root.subtree)
            if has_subordinate:
                complex_tunits += 1
        features['num_complex_tunits'] = float(complex_tunits)
        
        # 并列短语
        coordinate_phrases = [t for t in doc if t.dep_ == 'conj']
        features['num_coordinate_phrases'] = float(len(coordinate_phrases))
        
        # 复杂名词短语（名词+修饰语）
        complex_nominals = [t for t in doc if t.pos_ == 'NOUN' and any(child.dep_ in {'amod', 'nmod', 'acl'} for child in t.children)]
        features['num_complex_nominals'] = float(len(complex_nominals))
        
        # 动词短语
        verb_phrases = [t for t in doc if t.pos_ == 'VERB']
        features['num_verb_phrases'] = float(len(verb_phrases))
        
        # 比率特征
        w = features['num_words']
        s = features['num_sentences'] if features['num_sentences'] > 0 else 1
        c = features['num_clauses']
        dc = features['num_dependent_clauses']
        t = features['num_tunits'] if features['num_tunits'] > 0 else 1
        ct = features['num_complex_tunits']
        cp = features['num_coordinate_phrases']
        cn = features['num_complex_nominals']
        vp = features['num_verb_phrases']
        
        features['mean_length_sentence'] = w / s
        features['mean_length_tunit'] = w / t
        features['sentence_complexity'] = c / s
        features['subordination_ratio_C_T'] = c / t
        features['subordination_ratio_CT_T'] = ct / t
        features['subordination_ratio_DC_C'] = dc / c if c > 0 else 0.0
        features['subordination_ratio_DC_T'] = dc / t
        features['coordination_ratio_CP_C'] = cp / c if c > 0 else 0.0
        features['coordination_ratio_CP_T'] = cp / t
        features['tunit_per_sentence'] = t / s
        features['phrasal_sophistication_CN_C'] = cn / c if c > 0 else 0.0
        features['phrasal_sophistication_CN_T'] = cn / t
        features['phrasal_sophistication_VP_T'] = vp / t
        
        return features
    
    def _extract_syntactic_features_rules(self, query: str) -> Dict[str, float]:
        """使用规则方法提取句法结构特征（轻量级）"""
        features = {}
        
        tokens = query.lower().split()
        tokens = [t.strip('.,!?;:') for t in tokens if t.strip()]
        
        # 基础计数
        features['num_words'] = float(len(tokens))
        features['num_sentences'] = 1.0  # 简化
        
        # 从句标记
        clause_indicators = ['that', 'which', 'who', 'where', 'when', 'if', 'because', 'although', 'while']
        num_clauses = sum(1 for t in tokens if t in clause_indicators)
        features['num_clauses'] = float(num_clauses)
        features['num_dependent_clauses'] = float(num_clauses)
        
        # T-units（简化为主句数）
        features['num_tunits'] = 1.0
        features['num_complex_tunits'] = float(min(1, num_clauses))
        
        # 并列（and, or, but）
        coord_words = {'and', 'or', 'but', 'nor'}
        features['num_coordinate_phrases'] = float(sum(1 for t in tokens if t in coord_words))
        
        # 名词（大写开头的词）
        words = query.split()
        features['num_complex_nominals'] = float(sum(1 for w in words if w[0].isupper()))
        
        # 动词
        common_verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        features['num_verb_phrases'] = float(sum(1 for t in tokens if t in common_verbs))
        
        # 比率特征
        w = features['num_words']
        s = features['num_sentences']
        c = features['num_clauses'] if features['num_clauses'] > 0 else 1
        t = features['num_tunits'] if features['num_tunits'] > 0 else 1
        
        features['mean_length_sentence'] = w / s
        features['mean_length_tunit'] = w / t
        features['sentence_complexity'] = c / s
        features['subordination_ratio_C_T'] = c / t
        features['subordination_ratio_CT_T'] = features['num_complex_tunits'] / t
        features['subordination_ratio_DC_C'] = features['num_dependent_clauses'] / c
        features['subordination_ratio_DC_T'] = features['num_dependent_clauses'] / t
        features['coordination_ratio_CP_C'] = features['num_coordinate_phrases'] / c
        features['coordination_ratio_CP_T'] = features['num_coordinate_phrases'] / t
        features['tunit_per_sentence'] = t / s
        features['phrasal_sophistication_CN_C'] = features['num_complex_nominals'] / c
        features['phrasal_sophistication_CN_T'] = features['num_complex_nominals'] / t
        features['phrasal_sophistication_VP_T'] = features['num_verb_phrases'] / t
        
        return features
    
    # ========== 2. Dependency-based Structural Features ==========
    
    def _extract_dependency_features(self, doc) -> Dict[str, float]:
        """使用spaCy提取依存结构特征"""
        features = {}
        
        # 计算依存距离
        distances = []
        for token in doc:
            if token.head != token:
                distance = abs(token.i - token.head.i)
                distances.append(distance)
        
        if distances:
            features['max_dependency_distance'] = float(max(distances))
            features['avg_dependency_distance'] = float(np.mean(distances))
            features['num_long_range_dependencies'] = float(sum(1 for d in distances if d > 5))
        else:
            features['max_dependency_distance'] = 0.0
            features['avg_dependency_distance'] = 0.0
            features['num_long_range_dependencies'] = 0.0
        
        # 关系类型计数
        features['num_subject_verb'] = float(len([t for t in doc if t.dep_ in {'nsubj', 'nsubjpass'}]))
        features['num_object_verb'] = float(len([t for t in doc if t.dep_ in {'dobj', 'iobj', 'obj'}]))
        features['num_modifier'] = float(len([t for t in doc if t.dep_ in {'amod', 'nmod', 'advmod'}]))
        features['num_coordination_dep'] = float(len([t for t in doc if t.dep_ == 'conj']))
        features['num_subordination_dep'] = float(len([t for t in doc if t.dep_ in {'advcl', 'ccomp', 'mark'}]))
        
        # 树不平衡度（简化计算）
        depths = []
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            depths.append(depth)
        
        if depths:
            features['tree_imbalance'] = float(np.std(depths))
        else:
            features['tree_imbalance'] = 0.0
        
        return features
    
    def _extract_dependency_features_rules(self, query: str) -> Dict[str, float]:
        """使用规则方法提取依存特征"""
        features = {}
        
        tokens = query.split()
        
        # 简化估计
        features['max_dependency_distance'] = min(5.0, len(tokens) / 2)
        features['avg_dependency_distance'] = min(3.0, len(tokens) / 4)
        features['num_long_range_dependencies'] = 0.0
        
        features['num_subject_verb'] = 1.0  # 至少一个主谓关系
        features['num_object_verb'] = 0.5
        features['num_modifier'] = float(sum(1 for t in tokens if t.endswith('ly') or t.endswith('ful')))
        features['num_coordination_dep'] = float(sum(1 for t in tokens if t.lower() in {'and', 'or', 'but'}))
        features['num_subordination_dep'] = 0.0
        
        features['tree_imbalance'] = 1.0
        
        return features
    
    # ========== 3. Semantic and Lexical Features ==========
    
    def _extract_semantic_features(self, doc) -> Dict[str, float]:
        """使用spaCy提取语义词汇特征"""
        features = {}
        
        # 命名实体特征
        entities = list(doc.ents)
        features['num_entities'] = float(len(entities))
        
        entity_type_counts = Counter([ent.label_ for ent in entities])
        features['num_person_entities'] = float(entity_type_counts.get('PERSON', 0))
        features['num_org_entities'] = float(entity_type_counts.get('ORG', 0))
        features['num_location_entities'] = float(entity_type_counts.get('GPE', 0) + entity_type_counts.get('LOC', 0))
        features['num_date_entities'] = float(entity_type_counts.get('DATE', 0))
        
        tokens = [t for t in doc if not t.is_punct]
        features['entity_density'] = features['num_entities'] / len(tokens) if tokens else 0.0
        
        # 语义角色特征（简化版：使用依存关系近似）
        features['num_agents'] = float(len([t for t in doc if t.dep_ in {'nsubj', 'nsubjpass'}]))
        features['num_patients'] = float(len([t for t in doc if t.dep_ in {'dobj', 'iobj', 'obj'}]))
        features['num_temporal'] = float(len([t for t in doc if t.ent_type_ == 'DATE']))
        features['num_locative'] = float(len([t for t in doc if t.ent_type_ in {'GPE', 'LOC'}]))
        
        # 词汇多样性
        lemmas = [t.lemma_.lower() for t in tokens]
        features['lexical_diversity'] = len(set(lemmas)) / len(lemmas) if lemmas else 0.0
        
        # Content-to-function ratio
        content_tokens = [t for t in tokens if t.pos_ not in self.function_pos]
        features['content_function_ratio'] = len(content_tokens) / len(tokens) if tokens else 0.0
        
        # 信息密度
        features['information_density'] = len(content_tokens) / len(tokens) if tokens else 0.0
        
        # 问题类型
        tokens_lower = [t.text.lower() for t in tokens]
        question_words_in_query = set(tokens_lower) & self.question_words
        features['is_question_word'] = float(len(question_words_in_query) > 0)
        
        simple_qw = {'what', 'who', 'which', 'where', 'when'}
        complex_qw = {'how', 'why'}
        features['is_simple_question'] = float(bool(set(tokens_lower) & simple_qw))
        features['is_complex_question'] = float(bool(set(tokens_lower) & complex_qw))
        
        # 复杂度标记
        features['has_coordination'] = float(any(t.dep_ == 'conj' for t in doc))
        features['has_subordination'] = float(any(t.dep_ in {'advcl', 'ccomp', 'mark'} for t in doc))
        features['has_negation'] = float(any(t.text.lower() in self.negation_words for t in doc))
        features['has_passive_voice'] = float(any(t.dep_ == 'nsubjpass' for t in doc))
        
        return features
    
    def _extract_semantic_features_rules(self, query: str) -> Dict[str, float]:
        """使用规则方法提取语义特征"""
        features = {}
        
        tokens = query.lower().split()
        tokens_clean = [t.strip('.,!?;:') for t in tokens if t.strip()]
        
        # 命名实体（大写开头的词）
        words = query.split()
        entities = [w for w in words if w[0].isupper() and w.lower() not in ['the', 'a', 'an']]
        features['num_entities'] = float(len(entities))
        features['num_person_entities'] = 0.0  # 无法准确识别
        features['num_org_entities'] = 0.0
        features['num_location_entities'] = float(len(entities) * 0.5)  # 粗略估计
        features['num_date_entities'] = 0.0
        
        features['entity_density'] = features['num_entities'] / len(tokens_clean) if tokens_clean else 0.0
        
        features['num_agents'] = 1.0
        features['num_patients'] = 0.5
        features['num_temporal'] = 0.0
        features['num_locative'] = 0.0
        
        # 词汇多样性
        features['lexical_diversity'] = len(set(tokens_clean)) / len(tokens_clean) if tokens_clean else 0.0
        features['content_function_ratio'] = 0.7
        features['information_density'] = 0.7
        
        # 问题类型
        tokens_set = set(tokens_clean)
        features['is_question_word'] = float(bool(tokens_set & self.question_words))
        features['is_simple_question'] = float(bool(tokens_set & {'what', 'who', 'which', 'where', 'when'}))
        features['is_complex_question'] = float(bool(tokens_set & {'how', 'why'}))
        
        # 复杂度标记
        features['has_coordination'] = float(any(t in {'and', 'or', 'but'} for t in tokens_clean))
        features['has_subordination'] = float(any(t in {'if', 'because', 'although', 'while'} for t in tokens_clean))
        features['has_negation'] = float(any(t in self.negation_words for t in tokens_clean))
        features['has_passive_voice'] = 0.0
        
        return features
    
    # ========== 4. Tree Structure and Interaction Features ==========
    
    def _extract_tree_features(self, doc) -> Dict[str, float]:
        """使用spaCy提取树结构和交互特征"""
        features = {}
        
        # 计算树的深度和宽度
        def get_depth(token):
            if not list(token.children):
                return 1
            return 1 + max(get_depth(child) for child in token.children)
        
        roots = [t for t in doc if t.head == t]
        if roots:
            depths = [get_depth(root) for root in roots]
            features['max_tree_depth'] = float(max(depths))
        else:
            features['max_tree_depth'] = 1.0
        
        # 树宽度（每层的最大节点数）
        level_counts = {}
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            level_counts[depth] = level_counts.get(depth, 0) + 1
        
        features['max_tree_width'] = float(max(level_counts.values()) if level_counts else 1)
        
        # 叶节点与非叶节点比例
        leaves = sum(1 for t in doc if not list(t.children))
        non_leaves = len(list(doc)) - leaves
        features['leaf_nonleaf_ratio'] = leaves / non_leaves if non_leaves > 0 else 0.0
        
        # 分支因子
        if roots:
            branching_factors = []
            for root in roots:
                for token in root.subtree:
                    children = list(token.children)
                    if children:
                        branching_factors.append(len(children))
            features['branching_factor'] = float(np.mean(branching_factors)) if branching_factors else 0.0
        else:
            features['branching_factor'] = 0.0
        
        # 深度宽度比
        features['depth_width_ratio'] = features['max_tree_depth'] / features['max_tree_width'] if features['max_tree_width'] > 0 else 0.0
        
        # 交互特征
        tokens = [t for t in doc if not t.is_punct]
        clauses = [t for t in doc if t.dep_ in {'advcl', 'acl', 'relcl', 'csubj', 'ccomp'}]
        entities = list(doc.ents)
        
        features['tokens_per_clause'] = len(tokens) / len(clauses) if clauses else len(tokens)
        features['entities_per_token'] = len(entities) / len(tokens) if tokens else 0.0
        features['depth_per_token'] = features['max_tree_depth'] / len(tokens) if tokens else 0.0
        
        # 连接词数（coordinating conjunctions + subordination markers）
        connectors = [t for t in doc if t.dep_ in {'cc', 'mark'}]
        features['connectors_per_clause'] = len(connectors) / len(clauses) if clauses else 0.0
        
        # 额外交互特征
        features['complex_nominals_per_clause'] = features.get('num_complex_nominals', 0) / len(clauses) if clauses else 0.0
        features['verb_phrases_per_tunit'] = features.get('num_verb_phrases', 0) / max(1, len(roots))
        
        modifiers = [t for t in doc if t.dep_ in {'amod', 'nmod', 'advmod'}]
        features['modifiers_per_token'] = len(modifiers) / len(tokens) if tokens else 0.0
        
        return features
    
    def _extract_tree_features_rules(self, query: str) -> Dict[str, float]:
        """使用规则方法提取树特征"""
        features = {}
        
        tokens = query.split()
        
        features['max_tree_depth'] = min(5.0, len(tokens) / 3)
        features['max_tree_width'] = min(4.0, len(tokens) / 5)
        features['leaf_nonleaf_ratio'] = 2.0
        features['branching_factor'] = 1.5
        features['depth_width_ratio'] = features['max_tree_depth'] / features['max_tree_width']
        
        clauses = sum(1 for t in tokens if t.lower() in ['that', 'which', 'who', 'where', 'when', 'if'])
        features['tokens_per_clause'] = len(tokens) / max(1, clauses)
        features['entities_per_token'] = 0.1
        features['depth_per_token'] = features['max_tree_depth'] / len(tokens)
        features['connectors_per_clause'] = 0.5
        features['complex_nominals_per_clause'] = 0.5
        features['verb_phrases_per_tunit'] = 1.0
        features['modifiers_per_token'] = 0.1
        
        return features
    
    def _normalize_features(self, features_array: np.ndarray) -> np.ndarray:
        """
        归一化特征
        
        Args:
            features_array: 特征数组 (batch_size, num_features)
            
        Returns:
            归一化后的特征数组
        """
        if self.feature_stats is not None:
            # 使用提供的统计信息进行归一化
            for i, feat_name in enumerate(self.feature_names):
                if feat_name in self.feature_stats:
                    mean, std = self.feature_stats[feat_name]
                    if std > 1e-8:  # 避免除零
                        features_array[:, i] = (features_array[:, i] - mean) / std
        else:
            # 使用batch内的统计信息进行归一化（z-score）
            mean = features_array.mean(axis=0)
            std = features_array.std(axis=0) + 1e-8
            features_array = (features_array - mean) / std
        
        return features_array
    
    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        if self.selected_features is not None:
            return len(self.selected_features)
        return len(self.feature_names)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        if self.selected_features is not None:
            return self.selected_features.copy()
        return self.feature_names.copy()


if __name__ == '__main__':
    # 测试代码
    extractor = HandcraftedFeatureExtractor(use_spacy=False, normalize=False)
    
    test_queries = [
        "What is the capital of France?",
        "How does photosynthesis work in plants?",
        "Which movie starring Tom Hanks was released in 1994?",
        "If it rains tomorrow, will the match be cancelled?",
        "Who is the best player compared to Messi?"
    ]
    
    print("=" * 80)
    print("手工特征提取测试（基于RouteRAG论文）")
    print("=" * 80)
    
    for query in test_queries:
        features = extractor.extract_features(query)
        print(f"\nQuery: {query}")
        print(f"特征数量: {len(features)}")
        print(f"前10个特征:")
        for i, (name, value) in enumerate(list(features.items())[:10]):
            print(f"  {name:35s}: {value:.4f}")
    
    # 测试批量提取
    print("\n" + "=" * 80)
    print("批量提取测试")
    print("=" * 80)
    batch_features = extractor.extract_batch(test_queries)
    print(f"Batch shape: {batch_features.shape}")
    print(f"Feature dimension: {extractor.get_feature_dimension()}")
    print(f"Feature names count: {len(extractor.get_feature_names())}")
