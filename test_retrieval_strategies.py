#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同的GraphRAG检索策略
调整local_search和global_search的参数，观察性能变化
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径（指向routing_rag目录）
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 设置环境变量
os.environ['GRAPHRAG_API_KEY'] = os.getenv('GRAPHRAG_API_KEY', 'ollama')


def load_test_data(data_path: str, num_samples: int = 3) -> List[Dict]:
    """加载测试数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data[:num_samples]


def run_local_search_with_params(
    query: str,
    data_path: str,
    config_file: str = None,
    top_k_related_entities: int = 10,
    top_k_neighbors: int = 10,
    max_tokens: int = 4000,
    response_type: str = 'single paragraph'
) -> Dict[str, Any]:
    """
    运行local_search并支持自定义参数

    Args:
        query: 查询字符串
        data_path: 数据目录路径
        config_file: 配置文件名（可选）
        top_k_related_entities: 返回的相关实体数量
        top_k_neighbors: 每个实体的邻居数量
        max_tokens: 上下文的最大token数
        response_type: 响应类型

    Returns:
        包含检索结果的字典
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
        vector_size=768
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

    # 注意：这里我们无法直接传递top_k等参数
    # 因为get_local_search_engine不支持这些参数
    # 这些参数是在local_search的配置文件中定义的
    # 我们可以修改配置文件来测试不同的参数

    search_engine = get_local_search_engine(
        config=config,
        reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates={},
        response_type=response_type,
        description_embedding_store=entity_description_embedding_store,
        system_prompt=system_prompt
    )

    # 执行查询
    start_time = time.time()
    result = asyncio.run(search_engine.search(query=query))
    end_time = time.time()

    entities_result = result.context_data.get('entities', [])
    relationships_result = result.context_data.get('relationships', [])
    sources_result = result.context_data.get('sources', [])

    return {
        'query': query,
        'response': result.response,
        'context_text_length': len(result.context_text) if result.context_text else 0,
        'num_entities': len(entities_result),
        'num_relationships': len(relationships_result),
        'num_sources': len(sources_result),
        'completion_time': end_time - start_time,
        'params': {
            'top_k_related_entities': top_k_related_entities,
            'top_k_neighbors': top_k_neighbors,
            'max_tokens': max_tokens,
            'response_type': response_type
        }
    }


def test_different_strategies(
    query: str,
    data_path: str,
    config_file: str = None
) -> Dict[str, Any]:
    """
    测试不同的检索策略

    Args:
        query: 查询字符串
        data_path: 数据目录路径
        config_file: 配置文件名（可选）

    Returns:
        测试结果字典
    """
    strategies = [
        {
            'name': 'Strategy 1: Conservative (small context)',
            'params': {
                'top_k_related_entities': 5,
                'top_k_neighbors': 5,
                'max_tokens': 2000,
                'response_type': 'single paragraph'
            }
        },
        {
            'name': 'Strategy 2: Balanced (default)',
            'params': {
                'top_k_related_entities': 10,
                'top_k_neighbors': 10,
                'max_tokens': 4000,
                'response_type': 'single paragraph'
            }
        },
        {
            'name': 'Strategy 3: Aggressive (large context)',
            'params': {
                'top_k_related_entities': 20,
                'top_k_neighbors': 20,
                'max_tokens': 8000,
                'response_type': 'single paragraph'
            }
        }
    ]

    # 注意：当前GraphRAG API不支持直接传递这些参数
    # 这些参数需要在配置文件中设置
    # 所以这里我们只能测试response_type参数

    response_types = [
        {
            'name': 'Response Type: Single Paragraph',
            'params': {
                'response_type': 'single paragraph'
            }
        },
        {
            'name': 'Response Type: Multiple Paragraphs',
            'params': {
                'response_type': 'multiple paragraphs'
            }
        },
        {
            'name': 'Response Type: List',
            'params': {
                'response_type': 'list'
            }
        }
    ]

    all_results = {}

    for strategy in response_types:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy['name']}")
        print(f"{'='*60}")

        result = run_local_search_with_params(
            query=query,
            data_path=data_path,
            config_file=config_file,
            response_type=strategy['params']['response_type']
        )

        print(f"\nResponse: {result['response']}")
        print(f"Context length: {result['context_text_length']} chars")
        print(f"Retrieved: {result['num_entities']} entities, "
              f"{result['num_relationships']} relationships, "
              f"{result['num_sources']} sources")
        print(f"Time: {result['completion_time']:.2f}s")

        all_results[strategy['name']] = result

    return all_results


def main():
    """
    主函数
    """
    # 选择要测试的索引
    index_path = os.path.join(project_root, 'graphrag_ollama_hotpotqa_test_data')
    config_file = 'graphrag_hotpotqa_config.yml'

    # 加载测试数据
    test_data_path = os.path.join(project_root, 'HotpotQA', 'hotpot_1000_samples.jsonl')
    test_samples = load_test_data(test_data_path, num_samples=3)

    print(f"\nLoaded {len(test_samples)} test samples")
    print(f"Testing on index: {index_path}")

    # 对每个query测试不同的策略
    all_strategy_results = {}

    for i, sample in enumerate(test_samples):
        query = sample['question']
        print(f"\n\n{'#'*80}")
        print(f"# Query {i+1}/{len(test_samples)}")
        print(f"{'#'*80}")
        print(f"Question: {query}")
        print(f"Answer: {sample['answer']}")

        # 测试不同的策略
        strategy_results = test_different_strategies(
            query=query,
            data_path=index_path,
            config_file=config_file
        )

        # 添加ground truth
        strategy_results['ground_truth'] = sample['answer']
        all_strategy_results[f'query_{i+1}'] = strategy_results

        # 暂停让用户查看结果
        input("\nPress Enter to continue to next query...")

    # 保存结果
    output_dir = os.path.join(project_root, 'strategy_test_results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'strategy_test_{timestamp}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_strategy_results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*60}")
    print(f"Strategy test results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
