"""
Graph RAG Implementation using Microsoft GraphRAG
基于微软GraphRAG的图增强检索实现
"""
from typing import Dict, Any
from interfaces.rag_interface import RAGInterface
from config.config import settings
import logging
import sys
import os

# 添加GraphRAG库到路径
GRAPH_RAG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'graphrag')
GRAPH_RAG_PATH = os.path.abspath(GRAPH_RAG_PATH)
sys.path.insert(0, GRAPH_RAG_PATH)


class GraphRAG(RAGInterface):
    """
    基于微软GraphRAG的图增强检索实现
    Graph-augmented Retrieval implementation using Microsoft GraphRAG
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化GraphRAG

        Args:
            config: 配置参数字典（可选）
            
        注意：GraphRAG的实际查询行为由独立的.yml配置文件控制（如graphrag_hotpotqa_config.yml）
        settings.yaml中的graph_rag配置不会被使用
        """
        self.config = config or {}

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 记录最后一次检索和生成的时间（用于性能评测）
        # 注意：如果 SearchResult 有精确的 retrieval_time 字段，则使用精确值；
        # 否则使用经验比例估算：检索 75%，生成 25%
        self.last_retrieval_time = 0.0
        self.last_generation_time = 0.0

        # 搜索引擎实例及其就绪标志（延迟初始化，首次 execute() 时创建）
        self._search_engine = None
        self._search_engine_ready = False

        # 尝试导入微软GraphRAG，如果不存在则后续处理
        try:
            # 检查GraphRAG模块是否可用
            import graphrag
            from graphrag.config.models.graph_rag_config import GraphRagConfig
            from graphrag.query.factory import (
                get_local_search_engine,
                get_global_search_engine,
                get_drift_search_engine,
                get_basic_search_engine
            )
            from graphrag.query.structured_search.local_search.search import LocalSearch
            from graphrag.query.structured_search.global_search.search import GlobalSearch
            from graphrag.data_model.community_report import CommunityReport
            from graphrag.data_model.text_unit import TextUnit
            from graphrag.data_model.entity import Entity
            from graphrag.data_model.relationship import Relationship
            from graphrag.data_model.covariate import Covariate
            from graphrag.vector_stores.base import BaseVectorStore

            self._graph_rag_available = True
            self.get_local_search_engine = get_local_search_engine
            self.get_global_search_engine = get_global_search_engine
            self.get_drift_search_engine = get_drift_search_engine
            self.get_basic_search_engine = get_basic_search_engine
            self.GraphRagConfig = GraphRagConfig
            self.LocalSearch = LocalSearch
            self.GlobalSearch = GlobalSearch
            self.CommunityReport = CommunityReport
            self.TextUnit = TextUnit
            self.Entity = Entity
            self.Relationship = Relationship
            self.Covariate = Covariate
            self.BaseVectorStore = BaseVectorStore

        except ImportError as e:
            self._graph_rag_available = False
            self.logger.warning(f"Microsoft GraphRAG not available: {str(e)}")
            self.logger.warning("Please ensure GraphRAG is properly installed in your environment")

    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        执行Graph RAG查询
        Execute Graph RAG query

        Args:
            query (str): 查询字符串
            context (Dict[str, Any], optional): 上下文信息，如图数据、实体、关系等

        Returns:
            str: 查询结果
        """
        if not self._graph_rag_available:
            return f"错误：Microsoft GraphRAG库不可用。无法执行查询: {query}"

        try:
            # 从上下文中获取必要参数
            search_mode = context.get('search_mode', 'local') if context else 'local'
            data_path = context.get('data_path', None) if context else None

            if not data_path:
                return f"错误：需要提供包含已索引数据的路径。请在context中指定'data_path'参数。"

            # 根据指定的搜索模式执行查询
            if search_mode == 'local':
                # print("使用本地搜索模式")
                return self._local_search(query, data_path, context)
            else:
                # 暂时只支持本地搜索，其他模式返回提示
                return f"当前仅支持本地搜索模式。查询: {query}"

        except Exception as e:
            self.logger.error(f"执行Graph RAG查询时出错: {str(e)}")
            return f"错误：执行Graph RAG查询时出现问题 - {str(e)}"

    def _has_graph_data(self, context: Dict[str, Any]) -> bool:
        """
        检查上下文是否包含图数据
        Check if context contains graph data
        """
        if not context:
            return False
        # 检查是否包含GraphRAG所需的关键数据结构
        required_keys = ['entities', 'relationships', 'reports', 'text_units', 'communities']
        return any(key in context for key in required_keys)

    def _execute_with_graph_data(self, query: str, graph_data: Dict[str, Any]) -> str:
        """
        使用图数据执行查询
        Execute query with graph data
        """
        # 这是一个简化实现，实际GraphRAG需要完整的索引数据
        # 包括实体、关系、社区报告等复杂结构

        # 模拟GraphRAG查询结果
        return f"GraphRAG已处理查询: '{query}'，使用图数据进行增强检索。"

    def build_index_from_path(self, root_dir: str, config_filepath: str = None, output_dir: str = None, method: str = None, **kwargs):
        """
        从路径构建索引（适用于文件系统驱动的RAG）
        Build index from file path (suitable for file system-based RAG)

        Args:
            root_dir (str): 项目根目录路径，配置文件中的相对路径将相对于此目录解析
            config_filepath (str, optional): 配置文件路径
            output_dir (str, optional): 输出目录路径，可覆盖配置文件中的设置
            method (str, optional): 索引构建方法，可选值: 'fast', 'standard'。默认为 'standard'
                - 'fast': 快速模式，使用NLP提取实体和关系，减少LLM调用
                - 'standard': 标准模式，使用LLM进行完整的图构建
            **kwargs: 额外的参数，用于特定实现的配置
        """
        if not self._graph_rag_available:
            self.logger.error("GraphRAG不可用，无法构建索引")
            return False

        try:
            from graphrag.cli.index import index_cli
            from graphrag.config.enums import IndexingMethod
            from pathlib import Path

            # 确保提供了有效的配置文件
            if config_filepath is None:
                self.logger.error("必须提供配置文件路径")
                return False

            # 验证根目录存在
            if not Path(root_dir).exists():
                self.logger.error(f"根目录不存在: {root_dir}")
                return False

            # 解析索引方法
            # 支持的method值: 'fast', 'standard'，不区分大小写
            indexing_method = IndexingMethod.Standard  # 默认使用标准模式
            if method:
                method_lower = method.lower()
                if method_lower == 'fast':
                    indexing_method = IndexingMethod.Fast
                    self.logger.info("使用Fast模式构建索引（NLP + LLM混合模式）")
                elif method_lower == 'standard':
                    indexing_method = IndexingMethod.Standard
                    self.logger.info("使用Standard模式构建索引（完整LLM模式）")
                else:
                    self.logger.warning(f"未知的索引方法 '{method}'，使用默认的Standard模式")

            # 使用CLI接口构建索引
            # root_dir参数是项目根目录，配置文件中的相对路径将相对于此目录解析
            index_cli(
                root_dir=Path(root_dir),
                verbose=True,
                memprofile=False,
                cache=True,
                config_filepath=Path(config_filepath),
                dry_run=False,
                skip_validation=False,
                # output_dir=Path(output_dir) if output_dir else None,
                output_dir=None,
                method=indexing_method
            )

            self.logger.info("GraphRAG索引构建成功")
            return True

        except Exception as e:
            self.logger.error(f"构建GraphRAG索引时出错: {str(e)}")
            return False

    def build_index_from_data(self, data, metadata=None, **kwargs):
        """
        从数据列表构建索引（适用于内存驱动的RAG）
        Build index from data list (suitable for in-memory RAG)

        GraphRAG需要数据在文件系统中，不支持直接从数据列表构建索引。
        GraphRAG requires data to be in file system, does not support building index directly from data list.

        Args:
            data (List[str]): 用于构建索引的文档数据列表
            metadata (Optional[List[Dict[str, Any]]]): 与文档关联的元数据列表
            **kwargs: 额外的参数，用于特定实现的配置

        Returns:
            bool: 总是返回False，因为GraphRAG不支持此功能
        """
        self.logger.error("GraphRAG requires data to be in file system. Please use build_index_from_path().")
        return False

    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """
        添加文档到图索引（简化实现）
        Add a document to the graph index (simplified implementation)

        Args:
            text (str): 文档文本
            metadata (Dict[str, Any], optional): 元数据
        """
        if not self._graph_rag_available:
            self.logger.error("GraphRAG不可用，无法添加文档")
            return

        # 在完整实现中，这将涉及到实体提取、关系识别、社区发现等复杂流程
        self.logger.info("GraphRAG添加文档功能需要完整的图构建流程，当前为简化实现")

    def _find_config_file(self, data_dir) -> str:
        """
        在指定目录下查找GraphRAG配置文件
        Find GraphRAG config file in the specified directory

        Args:
            data_dir: 数据目录路径

        Returns:
            配置文件路径，如果未找到则返回None
        """
        from pathlib import Path

        # 常见的配置文件名列表（按优先级排序）
        config_file_names = [
            'config.yml',
            'settings.yml',
            'graphrag_config.yml',
            'graphrag.yml',
        ]

        for config_name in config_file_names:
            config_path = data_dir / config_name
            if config_path.exists():
                self.logger.info(f"找到配置文件: {config_path}")
                return str(config_path)

        return None

    def _get_vector_store_schema(self, config, context: Dict[str, Any] = None):
        """
        获取向量存储schema配置
        Get vector store schema configuration

        Args:
            config: GraphRAG配置对象
            context: 上下文信息，可包含自定义的向量存储配置

        Returns:
            VectorStoreSchemaConfig对象
            
        维度确定优先级：
        1. embedding_dim 配置（优先级：context > self.config）
        2. 从 embedding_model 名推断（必须在 dimension_map 中）
        3. 推断失败 → 报错并退出
        """
        from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig

        # 准备合并配置
        # 字典解包合并规则：后面的字典覆盖前面的字典
        # 因此优先级：context > self.config
        merged_config = {**self.config, **(context or {})}
        
        # 1. 尝试从配置中获取显式的维度
        dimensions = None
        if 'embedding_dim' in merged_config:
            dimensions = merged_config['embedding_dim']
        
        # 3. 如果没有显式配置，尝试从 embedding_model 推断
        if dimensions is None:
            # 从 GraphRAG 的 config 对象中提取 embedding_model（来自.yml配置文件）
            embedding_model = None
            if hasattr(config, 'models'):
                for model_id, model_config in config.models.items():
                    if hasattr(model_config, 'type') and model_config.type == 'embedding':
                        embedding_model = model_config.model
                        break
            
            # 优先使用 merged_config 中的 embedding_model（如果通过context传递）
            embedding_model = merged_config.get('embedding_model', embedding_model)
            
            # 根据模型推断维度
            # 支持 Ollama 格式 (如 nomic-embed-text) 和 vLLM/HuggingFace 格式 (如 nomic-ai/nomic-embed-text-v1)
            dimension_map = {
                # Ollama 格式
                'nomic-embed-text': 768,
                'mxbai-embed-large': 1024,
                'all-minilm': 384,
                # vLLM/HuggingFace 格式
                'nomic-ai/nomic-embed-text-v1': 768,
                'BAAI/bge-small-en-v1.5': 384,
                'BAAI/bge-base-en-v1.5': 768,
                'BAAI/bge-large-en-v1.5': 1024,
                # OpenAI 格式
                'text-embedding-ada-002': 1536,
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072,
            }
            
            if embedding_model and embedding_model in dimension_map:
                dimensions = dimension_map[embedding_model]
                self.logger.info(f"从embedding模型推断维度: {embedding_model} -> {dimensions}")
            else:
                # 既没有显式配置，模型又不在map中 → 严格报错
                raise ValueError(
                    f"无法确定embedding模型 '{embedding_model}' 的维度。\n"
                    f"请通过以下方式之一解决：\n"
                    f"  1. 在配置中显式指定 'embedding_dim' 参数\n"
                    f"  2. 或使用已知模型: {list(dimension_map.keys())}"
                )
        
        # 构建schema配置
        default_schema = {
            'index_name': 'default-entity-description',
            'id_field': 'id',
            'text_field': 'text',
            'vector_field': 'vector',
            'attributes_field': 'attributes',
            'vector_size': dimensions
        }
        
        # 如果有自定义schema，合并（后面的覆盖前面的）
        # 优先级：context['vector_store_schema'] > default_schema
        if context and 'vector_store_schema' in context:
            merged_schema = {**default_schema, **context['vector_store_schema']}
            self.logger.info(f"使用自定义向量存储schema配置，向量维度: {merged_schema['vector_size']}")
        else:
            merged_schema = default_schema
            self.logger.info(f"使用默认向量存储schema配置，向量维度: {dimensions}")

        # 创建VectorStoreSchemaConfig对象
        return VectorStoreSchemaConfig(**merged_schema)

    def _clean_row(self, row_dict: dict, string_keys: list) -> dict:
        """
        清理DataFrame行数据，处理NaN值和数据类型

        Args:
            row_dict: 行数据字典
            string_keys: 需要强制转为字符串的字段名列表

        Returns:
            清理后的字典
        """
        import numpy as np
        import pandas as pd
        cleaned_row = {}
        for key, value in row_dict.items():
            if isinstance(value, np.ndarray):
                cleaned_row[key] = value.tolist()
            elif pd.isna(value):
                cleaned_row[key] = None
            elif isinstance(value, (np.integer, int)):
                cleaned_row[key] = str(value) if key in string_keys else int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned_row[key] = float(value) if not pd.isna(value) else None
            else:
                cleaned_row[key] = value
        return cleaned_row

    def _init_search_engine(self, data_path: str, context: Dict[str, Any] = None) -> str:
        """
        初始化搜索引擎（数据加载、对象转换、向量库连接、引擎创建）。
        只需调用一次，之后通过 self._search_engine 复用。

        为什么拆出来：
          微软官方的 api/query.py 是一次性使用模式（每次查询都重建引擎），
          但 LocalSearch 类本身是无状态的，天然支持多次 search() 调用。
          评测场景下需要对同一个 GraphRAG 实例执行多次查询，
          如果每次都重复加载 parquet + DataFrame→对象转换 + 创建引擎，
          大量时间会浪费在重复初始化上（占总时间 ~90%）。

        Returns:
            成功返回空字符串 ""，失败返回非空的错误信息字符串。
            调用方通过 "if error:" 判断是否初始化成功。
        """
        if context is None:
            context = {}

        try:
            from graphrag.query.factory import get_local_search_engine
            from graphrag.config.load_config import load_config
            from graphrag.data_model.community_report import CommunityReport
            from graphrag.data_model.text_unit import TextUnit
            from graphrag.data_model.entity import Entity
            from graphrag.data_model.relationship import Relationship
            from graphrag.vector_stores.lancedb import LanceDBVectorStore
            from pathlib import Path
            import pandas as pd
            import numpy as np

            data_dir = Path(data_path)
            output_dir = data_dir / "output"

            if not output_dir.exists():
                return f"错误：输出目录不存在: {output_dir}"

            # 读取parquet文件
            entities_path = output_dir / "entities.parquet"
            relationships_path = output_dir / "relationships.parquet"
            reports_path = output_dir / "community_reports.parquet"
            text_units_path = output_dir / "text_units.parquet"

            for path, name in [(entities_path, "实体"), (relationships_path, "关系"),
                               (reports_path, "报告"), (text_units_path, "文本单元")]:
                if not path.exists():
                    return f"错误：{name}数据文件不存在: {path}"

            entities_df = pd.read_parquet(entities_path)
            relationships_df = pd.read_parquet(relationships_path)
            reports_df = pd.read_parquet(reports_path)
            text_units_df = pd.read_parquet(text_units_path)

            # 加载配置文件
            config_filename = context.get('config_filename', None)
            if config_filename:
                config_file_path = data_dir / config_filename
                if not config_file_path.exists():
                    return f"错误：指定的配置文件不存在: {config_file_path}"
                self.logger.info(f"使用指定的配置文件: {config_file_path}")
            else:
                config_file_path = self._find_config_file(data_dir)
                if not config_file_path:
                    return f"错误：未找到GraphRAG配置文件。尝试过的文件名: config.yml, settings.yml, graphrag_config.yml, graphrag.yml。请在context中指定'config_filename'参数。"
                self.logger.info(f"自动找到配置文件: {config_file_path}")

            config = load_config(root_dir=data_dir, config_filepath=Path(config_file_path))

            lancedb_path = output_dir / "lancedb"
            entity_description_vector_store_path = lancedb_path / "default-entity-description.lance"

            if not entity_description_vector_store_path.exists():
                return f"错误：未找到实体描述向量存储: {entity_description_vector_store_path}"

            # 将DataFrame转换为GraphRAG对象
            entity_string_keys = ['id', 'human_readable_id', 'title']
            entities = [
                Entity.from_dict(self._clean_row(row.to_dict(), entity_string_keys))
                for _, row in entities_df.iterrows()
            ]

            relationship_string_keys = ['id', 'human_readable_id', 'source', 'target']
            relationships = []
            for _, row in relationships_df.iterrows():
                cleaned = self._clean_row(row.to_dict(), relationship_string_keys)
                try:
                    relationships.append(Relationship.from_dict(cleaned))
                except Exception:
                    relationships.append(Relationship(
                        id=cleaned.get('id', ''), short_id=cleaned.get('human_readable_id'),
                        source=cleaned.get('source', ''), target=cleaned.get('target', ''),
                        description=cleaned.get('description', ''), rank=cleaned.get('rank', 1),
                        weight=cleaned.get('weight', 1.0), text_unit_ids=cleaned.get('text_unit_ids'),
                        attributes=cleaned.get('attributes')
                    ))

            report_string_keys = ['id', 'human_readable_id', 'title', 'community', 'community_id']
            reports = []
            for _, row in reports_df.iterrows():
                cleaned = self._clean_row(row.to_dict(), report_string_keys)
                try:
                    reports.append(CommunityReport.from_dict(cleaned))
                except Exception:
                    reports.append(CommunityReport(
                        id=cleaned.get('id', ''), title=cleaned.get('title', ''),
                        short_id=cleaned.get('human_readable_id'),
                        community_id=cleaned.get('community', ''),
                        summary=cleaned.get('summary', ''),
                        full_content=cleaned.get('full_content', ''),
                        rank=cleaned.get('rank', 1.0), attributes=cleaned.get('attributes'),
                        size=cleaned.get('size'), period=cleaned.get('period')
                    ))

            text_unit_string_keys = ['id', 'human_readable_id']
            text_units = []
            for _, row in text_units_df.iterrows():
                cleaned = self._clean_row(row.to_dict(), text_unit_string_keys)
                try:
                    text_units.append(TextUnit.from_dict(cleaned))
                except Exception:
                    text_units.append(TextUnit(
                        id=cleaned.get('id', ''), short_id=cleaned.get('human_readable_id'),
                        text=cleaned.get('text', ''), entity_ids=cleaned.get('entity_ids'),
                        relationship_ids=cleaned.get('relationship_ids'),
                        covariate_ids=cleaned.get('covariate_ids'),
                        n_tokens=cleaned.get('n_tokens'),
                        document_ids=cleaned.get('document_ids'),
                        attributes=cleaned.get('attributes')
                    ))

            # 初始化向量存储
            schema_config = self._get_vector_store_schema(config, context)
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

            self._search_engine = get_local_search_engine(
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

            # 初始化完成，标记为就绪（后续查询将直接复用 self._search_engine）
            self._search_engine_ready = True
            self.logger.info("搜索引擎初始化完成")
            return ""  # 成功：返回空字符串

        except Exception as e:
            self.logger.error(f"初始化搜索引擎时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"初始化搜索引擎时出错: {str(e)}"  # 失败：返回非空错误信息

    def _local_search(self, query: str, data_path: str, context: Dict[str, Any] = None) -> str:
        """
        本地搜索模式（初始化只执行一次，之后复用搜索引擎）

        采用延迟初始化（lazy init）模式：
          - 首次调用时，_search_engine_ready 为 False，执行完整的初始化流程
          - 后续调用时，直接复用已创建的 self._search_engine 实例
        """
        if context is None:
            context = {}

        # 延迟初始化搜索引擎：仅在第一次调用时执行
        # 后续调用时 _search_engine_ready 已为 True，直接跳过
        if not self._search_engine_ready:
            error = self._init_search_engine(data_path, context)
            if error:
                # 初始化失败，返回错误信息（error 为非空字符串）
                return error

        # 执行查询
        import asyncio
        import time

        total_start = time.time()
        try:
            if asyncio.iscoroutinefunction(self._search_engine.search):
                result = asyncio.run(self._search_engine.search(query=query))
            else:
                result = self._search_engine.search(query=query)

            total_end = time.time()
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            print(f"Retrieved Context Summary:")
            print(f"  - Entities: {len(result.context_data.get('entities', []))}")
            print(f"  - Relationships: {len(result.context_data.get('relationships', []))}")
            print(f"  - Sources: {len(result.context_data.get('sources', []))}")
            print(f"\nContext Text (first 2000 chars):")
            print(result.context_text[:2000] if len(result.context_text) > 2000 else result.context_text)
            context_text = result.context_text
            print(f"\nContext Text type: {type(context_text)}")
            print(f"Context Text length: {len(context_text) if context_text else 0}")
            print(f"Context Text repr: {repr(context_text[:200]) if context_text else 'None'}")
            print(f"\n{'='*60}")
            print(f"Response: {result.response}")
            print(f"Completion Time: {result.completion_time:.2f}s")
            print(f"{'='*60}\n")
            total_time = total_end - total_start

            # GraphRAG 的 search 方法包含了检索（图数据查询）和生成（LLM 回答）
            # SearchResult 现在包含精确的 retrieval_time 字段
            # retrieval_time = 嵌入生成时间 + 向量搜索时间 + 实体匹配后处理时间
            if hasattr(result, 'retrieval_time') and result.retrieval_time > 0:
                self.last_retrieval_time = result.retrieval_time
                self.last_generation_time = result.completion_time - result.retrieval_time
            else:
                # 如果没有 retrieval_time（向后兼容），使用经验比例估算
                retrieval_ratio = 0.75
                self.last_retrieval_time = total_time * retrieval_ratio
                self.last_generation_time = total_time * (1 - retrieval_ratio)

            if hasattr(result, 'completion_time'):
                total_time = result.completion_time

            # 最终输出: 检查 result 对象是否有 'response' 属性
            if hasattr(result, 'response'):
                return result.response
            else:
                self.logger.warning("Search result object does not have a 'response' attribute.")
                return "错误：无法从搜索结果中提取答案。"
        except Exception as e:
            self.logger.error(f"执行异步搜索时出错: {str(e)}")
            return f"执行异步搜索时出错: {str(e)}"