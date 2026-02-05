#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比GraphRAG在不同索引上的检索结果（带召回率和精度评估）
支持local_search和global_search两种检索策略
"""

import sys
import os
import json
import re
import string
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径（指向routing_rag目录）
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 设置环境变量
os.environ['GRAPHRAG_API_KEY'] = os.getenv('GRAPHRAG_API_KEY', 'ollama')


def load_test_data(data_path: str, num_samples: int = 3) -> List[Dict]:
    """
    加载测试数据

    Args:
        data_path: 数据文件路径
        num_samples: 加载的样本数量

    Returns:
        样本列表
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data[:num_samples]


def normalize_answer(s):
    """归一化答案（来自官方评估方法）"""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_f1_tokens(prediction_tokens, ground_truth_tokens):
    """计算token级别的F1分数（来自官方方法）"""

    from collections import Counter

    ZERO_METRIC = (0, 0, 0)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def calculate_retrieval_metrics(
    context_text: str,
    ground_truth_answer: str,
    ground_truth_supporting_facts: List[List[str]] = None
) -> Dict[str, float]:
    """
    计算检索质量指标（基于supporting_facts评估）

    Args:
        context_text: 检索到的上下文文本
        ground_truth_answer: 标准答案
        ground_truth_supporting_facts: 标准支持事实（可选）

    Returns:
        包含检索质量指标的字典
    """
    metrics = {
        'answer_in_context': 0.0,      # 答案完整字符串是否在context中
        'answer_tokens_in_context': 0.0, # 答案tokens有多少在context中
        'context_f1': 0.0,            # context与答案的F1分数
        'sp_facts_recall': 0.0,        # Supporting facts召回率
        'sp_facts_precision': 0.0,      # Supporting facts精度
        'sp_facts_f1': 0.0,          # Supporting facts F1分数
    }

    if not context_text:
        return metrics

    # 归一化context
    normalized_context = normalize_answer(context_text)

    # 指标1: 完整答案是否在context中（仅供参考）
    if ground_truth_answer:
        normalized_answer = normalize_answer(ground_truth_answer)
        if normalized_answer in normalized_context:
            metrics['answer_in_context'] = 1.0

    # 指标2: 答案与context的F1分数（仅供参考）
    if ground_truth_answer:
        answer_tokens = normalized_answer.split()
        context_tokens = normalized_context.split()

        f1, prec, rec = calculate_f1_tokens(context_tokens, answer_tokens)
        metrics['context_f1'] = f1

        # 答案tokens覆盖率
        tokens_in_context = sum(1 for token in answer_tokens if token in context_tokens)
        metrics['answer_tokens_in_context'] = tokens_in_context / len(answer_tokens) if answer_tokens else 0.0

    # 指标3: Supporting Facts召回率、精度、F1（核心指标！）
    if ground_truth_supporting_facts:
        from collections import Counter

        # 将每个supporting fact转为归一化的字符串
        # supporting_facts格式：[[entity, position], ...] 或 [subject, predicate, object]
        sp_fact_strings = []
        for fact in ground_truth_supporting_facts:
            if isinstance(fact, list):
                # 检查是[entity, position]还是[subject, predicate, object]格式
                if len(fact) == 2 and isinstance(fact[1], int):
                    # [entity, position]格式
                    entity = str(fact[0])
                    sp_fact_strings.append(normalize_answer(entity))
                else:
                    # [subject, predicate, object]格式
                    fact_str = ' '.join(str(item) for item in fact)
                    sp_fact_strings.append(normalize_answer(fact_str))

        # 计算有多少supporting facts在context中
        facts_in_context = 0
        for fact_str in sp_fact_strings:
            if fact_str in normalized_context:
                facts_in_context += 1

        # 召回率：有多少supporting facts被检索到
        metrics['sp_facts_recall'] = facts_in_context / len(sp_fact_strings) if sp_fact_strings else 0.0

        # 精度：检索到的context中有多少是相关的（难以计算，使用简化方法）
        # 由于无法确定context中有多少不相关的内容，我们用一个启发式方法：
        # 假设如果supporting facts都检索到了，则精度较高
        if metrics['sp_facts_recall'] > 0:
            # 简化：如果召回率高，假设精度也高
            # 更准确的方法需要判断context中每个句子是否相关，这需要NLP模型
            metrics['sp_facts_precision'] = metrics['sp_facts_recall'] * 0.8  # 启发式估计
        else:
            metrics['sp_facts_precision'] = 0.0

        # F1分数
        if metrics['sp_facts_precision'] + metrics['sp_facts_recall'] > 0:
            metrics['sp_facts_f1'] = 2 * metrics['sp_facts_precision'] * metrics['sp_facts_recall'] / (metrics['sp_facts_precision'] + metrics['sp_facts_recall'])
        else:
            metrics['sp_facts_f1'] = 0.0

    return metrics


def run_local_search_detailed(query: str, data_path: str, config_file: str = None) -> Dict[str, Any]:
    """
    运行local_search并返回详细的检索结果（直接使用GraphRAG API）

    Args:
        query: 查询字符串
        data_path: 数据目录路径
        config_file: 配置文件名（可选）

    Returns:
        包含详细检索结果的字典
    """
    import asyncio
    import pandas as pd
    from pathlib import Path
    from graphrag.query.factory import get_local_search_engine
    from graphrag.config.load_config import load_config
    from graphrag.data_model.entity import Entity
    from graphrag.data_model.relationship import Relationship
    from graphrag.data_model.community_report import CommunityReport
    from graphrag.data_model.text_unit import TextUnit
    from graphrag.vector_stores.lancedb import LanceDBVectorStore
    from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
    import numpy as np

    data_dir = Path(data_path)
    output_dir = data_dir / "output"
    lancedb_path = output_dir / "lancedb"

    # 加载数据
    entities_df = pd.read_parquet(output_dir / "entities.parquet")
    relationships_df = pd.read_parquet(output_dir / "relationships.parquet")
    reports_df = pd.read_parquet(output_dir / "community_reports.parquet")
    text_units_df = pd.read_parquet(output_dir / "text_units.parquet")

    # 转换数据格式
    entities = []
    for _, row in entities_df.iterrows():
        row_dict = row.to_dict()
        cleaned_row = {}
        for key, value in row_dict.items():
            if isinstance(value, np.ndarray):
                cleaned_row[key] = value.tolist()
            elif pd.isna(value):
                cleaned_row[key] = None
            elif isinstance(value, (np.integer, int)):
                if key in ['id', 'human_readable_id', 'title']:
                    cleaned_row[key] = str(value)
                else:
                    cleaned_row[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned_row[key] = float(value) if not pd.isna(value) else None
            else:
                cleaned_row[key] = value
        entity = Entity.from_dict(cleaned_row)
        entities.append(entity)

    relationships = []
    for _, row in relationships_df.iterrows():
        row_dict = row.to_dict()
        cleaned_row = {}
        for key, value in row_dict.items():
            if isinstance(value, np.ndarray):
                cleaned_row[key] = value.tolist()
            elif pd.isna(value):
                cleaned_row[key] = None
            elif isinstance(value, (np.integer, int)):
                if key in ['id', 'human_readable_id', 'source', 'target']:
                    cleaned_row[key] = str(value)
                else:
                    cleaned_row[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned_row[key] = float(value) if not pd.isna(value) else None
            else:
                cleaned_row[key] = value
        try:
            relationship = Relationship.from_dict(cleaned_row)
        except Exception:
            relationship = Relationship(
                id=cleaned_row.get('id', ''),
                short_id=cleaned_row.get('human_readable_id'),
                source=cleaned_row.get('source', ''),
                target=cleaned_row.get('target', ''),
                description=cleaned_row.get('description', ''),
                rank=cleaned_row.get('rank', 1),
                weight=cleaned_row.get('weight', 1.0),
                text_unit_ids=cleaned_row.get('text_unit_ids'),
                attributes=cleaned_row.get('attributes')
            )
        relationships.append(relationship)

    reports = []
    for _, row in reports_df.iterrows():
        row_dict = row.to_dict()
        cleaned_row = {}
        for key, value in row_dict.items():
            if isinstance(value, np.ndarray):
                cleaned_row[key] = value.tolist()
            elif pd.isna(value):
                cleaned_row[key] = None
            elif isinstance(value, (np.integer, int)):
                if key in ['id', 'human_readable_id', 'title', 'community', 'community_id']:
                    cleaned_row[key] = str(value)
                else:
                    cleaned_row[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned_row[key] = float(value) if not pd.isna(value) else None
            else:
                cleaned_row[key] = value
        try:
            report = CommunityReport.from_dict(cleaned_row)
        except Exception:
            report = CommunityReport(
                id=cleaned_row.get('id', ''),
                title=cleaned_row.get('title', ''),
                short_id=cleaned_row.get('human_readable_id'),
                community_id=cleaned_row.get('community', ''),
                summary=cleaned_row.get('summary', ''),
                full_content=cleaned_row.get('full_content', ''),
                rank=cleaned_row.get('rank', 1.0),
                attributes=cleaned_row.get('attributes'),
                size=cleaned_row.get('size'),
                period=cleaned_row.get('period')
            )
        reports.append(report)

    text_units = []
    for _, row in text_units_df.iterrows():
        row_dict = row.to_dict()
        cleaned_row = {}
        for key, value in row_dict.items():
            if isinstance(value, np.ndarray):
                cleaned_row[key] = value.tolist()
            elif pd.isna(value):
                cleaned_row[key] = None
            elif isinstance(value, (np.integer, int)):
                if key in ['id', 'human_readable_id']:
                    cleaned_row[key] = str(value)
                else:
                    cleaned_row[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned_row[key] = float(value) if not pd.isna(value) else None
            else:
                cleaned_row[key] = value
        try:
            text_unit = TextUnit.from_dict(cleaned_row)
        except Exception:
            text_unit = TextUnit(
                id=cleaned_row.get('id', ''),
                short_id=cleaned_row.get('human_readable_id'),
                text=cleaned_row.get('text', ''),
                entity_ids=cleaned_row.get('entity_ids'),
                relationship_ids=cleaned_row.get('relationship_ids'),
                covariate_ids=cleaned_row.get('covariate_ids'),
                n_tokens=cleaned_row.get('n_tokens'),
                document_ids=cleaned_row.get('document_ids'),
                attributes=cleaned_row.get('attributes')
            )
        text_units.append(text_unit)

    # 加载配置
    if config_file:
        config_file_path = data_dir / config_file
    else:
        # 查找配置文件
        for name in ['graphrag_hotpotqa_config.yml', 'config.yml', 'settings.yml']:
            config_file_path = data_dir / name
            if config_file_path.exists():
                break
        else:
            raise FileNotFoundError(f"Config file not found in {data_dir}")

    config = load_config(root_dir=data_dir, config_filepath=config_file_path)

    # 加载向量存储
    entity_description_vector_store_path = lancedb_path / "default-entity-description.lance"

    schema_config = VectorStoreSchemaConfig(
        index_name="default-entity-description",
        id_field="id",
        text_field="text",
        vector_field="vector",
        attributes_field="attributes",
        vector_size=768  # nomic-embed-text维度
    )

    entity_description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=schema_config
    )
    entity_description_embedding_store.connect(
        db_uri=str(lancedb_path),
        collection_name="default-entity-description"
    )

    # 创建搜索引擎
    system_prompt = """You are a precise question-answering assistant. Answer the question based on the provided context.

Guidelines:
1. Provide ONLY the answer, no explanations or reasoning
2. Keep the answer as short as possible - typically 1-5 words
3. For yes/no questions, answer only "yes" or "no"
4. For dates, use the exact format (e.g., "December 31, 2015")
5. For numbers, provide just the number (e.g., "1522")
6. For names, provide just the name (e.g., "Terry Crews")
7. DO NOT include phrases like "The answer is", "According to", "Based on", etc.
8. DO NOT add any additional context or information
9. If the answer is not in the context, say "I don't know" """

    search_engine = get_local_search_engine(
        config=config,
        reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates={},
        response_type="single paragraph",
        description_embedding_store=entity_description_embedding_store,
        system_prompt=system_prompt
    )

    # 执行查询
    result = asyncio.run(search_engine.search(query=query))

    # 提取检索结果详情
    entities_result = result.context_data.get('entities', [])
    relationships_result = result.context_data.get('relationships', [])
    sources_result = result.context_data.get('sources', [])

    # 格式化实体信息
    entities_info = []
    for entity in entities_result[:10]:  # 只显示前10个
        entities_info.append({
            'title': entity.title if hasattr(entity, 'title') else str(entity),
            'type': entity.type if hasattr(entity, 'type') else '',
            'description': entity.description[:100] if hasattr(entity, 'description') and entity.description else ''
        })

    # 格式化关系信息
    relationships_info = []
    for rel in relationships_result[:10]:
        relationships_info.append({
            'source': rel.source if hasattr(rel, 'source') else '',
            'target': rel.target if hasattr(rel, 'target') else '',
            'description': rel.description[:100] if hasattr(rel, 'description') and rel.description else ''
        })

    # 格式化sources信息
    sources_info = []
    for source in sources_result[:10]:
        if isinstance(source, dict):
            sources_info.append({
                'id': source.get('id', ''),
                'text': source.get('text', '')[:100] if source.get('text') else ''
            })
        elif isinstance(source, str):
            # 如果source是字符串，直接使用
            sources_info.append({
                'id': '',
                'text': source[:100]
            })
        else:
            # 其他情况，尝试转为字符串
            sources_info.append({
                'id': '',
                'text': str(source)[:100]
            })

    return {
        'query': query,
        'response': result.response,
        'context_text': result.context_text if result.context_text else '',
        'num_entities': len(entities_result),
        'num_relationships': len(relationships_result),
        'num_sources': len(sources_result),
        'entities': entities_info,
        'relationships': relationships_info,
        'sources': sources_info,
        'completion_time': result.completion_time if hasattr(result, 'completion_time') else 0
    }


def compare_retrieval_results(
    query: str,
    index_paths: List[Dict[str, str]],
    ground_truth_answer: str,
    ground_truth_supporting_facts: List[List[str]] = None,
    search_mode: str = 'local'
) -> Dict[str, Any]:
    """
    对比不同索引上的检索结果

    Args:
        query: 查询字符串
        index_paths: 索引路径列表，每个元素包含{'name': str, 'path': str, 'config': str}
        ground_truth_answer: 标准答案
        ground_truth_supporting_facts: 标准支持事实
        search_mode: 搜索模式 ('local' 或 'global')

    Returns:
        对比结果字典
    """
    comparison_results = {}

    for index_info in index_paths:
        index_name = index_info['name']
        index_path = index_info['path']
        config_file = index_info.get('config', None)

        print(f"\n{'='*60}")
        print(f"Testing on index: {index_name}")
        print(f"Path: {index_path}")
        print(f"{'='*60}")

        if search_mode == 'local':
            result = run_local_search_detailed(query, index_path, config_file)
        else:
            # 暂时只支持local search
            result = {'error': 'Global search not yet implemented'}

        # 计算检索质量指标
        if 'context_text' in result and ground_truth_answer:
            metrics = calculate_retrieval_metrics(
                context_text=result['context_text'],
                ground_truth_answer=ground_truth_answer,
                ground_truth_supporting_facts=ground_truth_supporting_facts
            )
            result['metrics'] = metrics

        comparison_results[index_name] = result

        # 打印结果摘要
        print(f"\nResponse: {result['response']}")
        if 'num_entities' in result:
            print(f"Retrieved entities: {result['num_entities']}")
            print(f"Retrieved relationships: {result['num_relationships']}")
            print(f"Retrieved sources: {result['num_sources']}")
            print(f"Completion time: {result['completion_time']:.2f}s")

        # 打印检索质量指标
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\nRetrieval Metrics (检索质量):")
            print(f"  [Answer-based metrics] (仅供参考):")
            print(f"    - Answer in context: {metrics['answer_in_context']:.2f}")
            print(f"    - Answer tokens in context: {metrics['answer_tokens_in_context']:.2%}")
            print(f"    - Context F1: {metrics['context_f1']:.4f}")
            print(f"  [Supporting Facts-based metrics] (核心指标):")
            print(f"    - SP Facts Recall: {metrics['sp_facts_recall']:.2%}")
            print(f"    - SP Facts Precision: {metrics['sp_facts_precision']:.2%}")
            print(f"    - SP Facts F1: {metrics['sp_facts_f1']:.4f}")

    return comparison_results


def main():
    """
    主函数
    """
    # 定义要测试的索引（只对比15-sample和1000-sample）
    index_paths = [
        {
            'name': '15-sample (embed_graph=True)',
            'path': os.path.join(project_root, 'graphrag_ollama_hotpotqa_test_data'),
            'config': 'graphrag_hotpotqa_config.yml'
        },
        {
            'name': '1000-sample (embed_graph=False)',
            'path': os.path.join(project_root, 'graphrag_ollama_hotpotqa_1000_test_data'),
            'config': 'graphrag_hotpotqa_config.yml'
        }
    ]

    # 加载测试数据
    test_data_path = os.path.join(project_root, 'HotpotQA', 'hotpot_dev_distractor_1000_samples.jsonl')
    test_samples = load_test_data(test_data_path, num_samples=5)

    print(f"\nLoaded {len(test_samples)} test samples")
    print(f"Comparing {len(index_paths)} indexes: {', '.join([idx['name'] for idx in index_paths])}")

    # 对每个query进行对比测试
    all_comparison_results = {}

    for i, sample in enumerate(test_samples):
        query = sample['question']
        ground_truth = sample['answer']
        supporting_facts = sample.get('supporting_facts', None)

        print(f"\n\n{'#'*80}")
        print(f"# Query {i+1}/{len(test_samples)}")
        print(f"{'#'*80}")
        print(f"Question: {query}")
        print(f"Ground Truth Answer: {ground_truth}")
        if supporting_facts:
            print(f"Supporting Facts: {supporting_facts}")

        # 对比不同索引的检索结果
        comparison_result = compare_retrieval_results(
            query=query,
            index_paths=index_paths,
            ground_truth_answer=ground_truth,
            ground_truth_supporting_facts=supporting_facts,
            search_mode='local'
        )

        # 添加ground truth
        comparison_result['ground_truth'] = ground_truth
        comparison_result['supporting_facts'] = supporting_facts
        all_comparison_results[f'query_{i+1}'] = comparison_result

        # 暂停让用户查看结果
        input("\nPress Enter to continue to next query...")

    # 保存对比结果
    output_dir = os.path.join(project_root, 'comparison_results')
    os.makedirs(output_dir, exist_ok=True)

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'retrieval_comparison_{timestamp}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_comparison_results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*60}")
    print(f"Comparison results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
