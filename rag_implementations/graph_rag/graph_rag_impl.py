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
            config: 配置参数字典
        """
        self.config = config or {}

        # 从配置中获取参数
        self.api_url = settings.graph_rag_api_url
        self.api_key = settings.graph_rag_api_key
        self.model = settings.graph_rag_model
        self.embedding_model = settings.graph_rag_embedding_model
        self.chunk_size = settings.graph_rag_chunk_size
        self.top_k = settings.graph_rag_top_k
        self.temperature = settings.graph_rag_temperature

        # 设置日志
        self.logger = logging.getLogger(__name__)

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
                return self._local_search(query, data_path)
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

    def build_index_from_path(self, root_dir: str, config_filepath: str = None, output_dir: str = None, **kwargs):
        """
        从路径构建索引（适用于文件系统驱动的RAG）
        Build index from file path (suitable for file system-based RAG)

        Args:
            root_dir (str): 项目根目录路径，配置文件中的相对路径将相对于此目录解析
            config_filepath (str, optional): 配置文件路径
            output_dir (str, optional): 输出目录路径，可覆盖配置文件中的设置
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
                method=IndexingMethod.Standard
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

    def _local_search(self, query: str, data_path: str) -> str:
        """
        本地搜索模式
        Local search mode
        """
        try:
            from graphrag.query.factory import get_local_search_engine
            from graphrag.config.models.graph_rag_config import GraphRagConfig
            from graphrag.config.load_config import load_config
            from pathlib import Path
            import pandas as pd
            import os

            # 要使用本地搜索，需要加载GraphRAG生成的数据
            data_dir = Path(data_path)
            output_dir = data_dir / "output"

            if not output_dir.exists():
                return f"错误：输出目录不存在: {output_dir}"

            # 尝试加载GraphRAG的索引数据
            # 读取实体、关系、报告和文本单元数据
            # 注意：根据实际生成的文件名进行调整
            entities_path = output_dir / "entities.parquet"
            relationships_path = output_dir / "relationships.parquet"
            reports_path = output_dir / "community_reports.parquet"
            text_units_path = output_dir / "text_units.parquet"

            # 检查必要的文件是否存在
            if not entities_path.exists():
                return f"错误：实体数据文件不存在: {entities_path}"
            if not relationships_path.exists():
                return f"错误：关系数据文件不存在: {relationships_path}"
            if not reports_path.exists():
                return f"错误：报告数据文件不存在: {reports_path}"
            if not text_units_path.exists():
                return f"错误：文本单元数据文件不存在: {text_units_path}"

            # 尝试加载这些parquet文件
            entities_df = pd.read_parquet(entities_path)
            relationships_df = pd.read_parquet(relationships_path)
            reports_df = pd.read_parquet(reports_path)
            text_units_df = pd.read_parquet(text_units_path)

            # 尝试加载配置文件（我们假设配置文件路径可以通过context传递）
            # 在实际使用中，用户需要提供索引构建时使用的配置文件路径
            config_file_path = data_dir / "graphrag_class_test_config.yml"
            if not config_file_path.exists():
                # 如果没有找到配置文件，返回错误信息
                # 在实际应用中，配置文件路径应该在context中提供
                return f"错误：未找到配置文件: {config_file_path}。请确保提供索引构建时使用的配置文件。"

            # 加载配置
            # 这里需要从配置文件目录作为根目录加载配置
            config_dir = data_dir  # 配置文件所在目录作为root_dir
            config = load_config(root_dir=Path(config_dir), config_filepath=Path(config_file_path))

            # 尝试加载LanceDB向量存储（GraphRAG使用）
            lancedb_path = output_dir / "lancedb"

            # 将pandas数据转换为GraphRAG所需的格式
            from graphrag.data_model.community_report import CommunityReport
            from graphrag.data_model.text_unit import TextUnit
            from graphrag.data_model.entity import Entity
            from graphrag.data_model.relationship import Relationship
            from graphrag.data_model.covariate import Covariate

            # 将DataFrame转换为相应对象列表（使用from_dict方法）
            import numpy as np
            entities = []
            for _, row in entities_df.iterrows():
                row_dict = row.to_dict()
                # 处理NaN值和数据类型，确保标识符字段是字符串
                # 对于Identified和Named基类要求的字段，需要确保是字符串类型
                cleaned_row = {}
                for key, value in row_dict.items():
                    if isinstance(value, np.ndarray):
                        cleaned_row[key] = value.tolist()  # 转换numpy数组
                    elif pd.isna(value):
                        cleaned_row[key] = None
                    elif isinstance(value, (np.integer, int)):
                        # 对于id、short_id（human_readable_id）和title字段，必须是字符串
                        if key in ['id', 'human_readable_id', 'title']:
                            cleaned_row[key] = str(value)
                        else:
                            # 其他数值字段可以保持数值类型
                            cleaned_row[key] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        cleaned_row[key] = float(value) if not pd.isna(value) else None
                    else:
                        cleaned_row[key] = value
                # 使用Entity的from_dict方法来正确构建对象
                entity = Entity.from_dict(cleaned_row)
                entities.append(entity)

            relationships = []
            reports = []
            text_units = []
            # 同样处理其他数据类型，但先简化处理避免复杂性
            # 对于其他类型，我们先使用基本构造方法
            from graphrag.data_model.relationship import Relationship
            from graphrag.data_model.community_report import CommunityReport
            from graphrag.data_model.text_unit import TextUnit

            # 为Relationship、CommunityReport和TextUnit也使用适当的方法
            import numpy as np
            for _, row in relationships_df.iterrows():
                row_dict = row.to_dict()
                # 处理NaN值和数据类型，确保标识符字段是字符串
                cleaned_row = {}
                for key, value in row_dict.items():
                    if isinstance(value, np.ndarray):
                        cleaned_row[key] = value.tolist()
                    elif pd.isna(value):
                        cleaned_row[key] = None
                    elif isinstance(value, (np.integer, int)):
                        # 对于id和human_readable_id字段，必须是字符串
                        if key in ['id', 'human_readable_id', 'source', 'target']:
                            cleaned_row[key] = str(value)
                        else:
                            cleaned_row[key] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        cleaned_row[key] = float(value) if not pd.isna(value) else None
                    else:
                        cleaned_row[key] = value
                # 使用Relationship的from_dict方法来正确构建对象
                try:
                    relationship = Relationship.from_dict(cleaned_row)
                except Exception:
                    # 如果from_dict失败，尝试手动构造
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

            for _, row in reports_df.iterrows():
                row_dict = row.to_dict()
                # 处理NaN值和数据类型，确保标识符字段是字符串
                cleaned_row = {}
                for key, value in row_dict.items():
                    if isinstance(value, np.ndarray):
                        cleaned_row[key] = value.tolist()
                    elif pd.isna(value):
                        cleaned_row[key] = None
                    elif isinstance(value, (np.integer, int)):
                        # 对于id、human_readable_id和title字段，必须是字符串
                        if key in ['id', 'human_readable_id', 'title', 'community', 'community_id']:
                            cleaned_row[key] = str(value)
                        else:
                            cleaned_row[key] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        cleaned_row[key] = float(value) if not pd.isna(value) else None
                    else:
                        cleaned_row[key] = value
                # 使用CommunityReport的from_dict方法来正确构建对象
                try:
                    report = CommunityReport.from_dict(cleaned_row)
                except Exception:
                    # 如果from_dict失败，尝试手动构造
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

            for _, row in text_units_df.iterrows():
                row_dict = row.to_dict()
                # 处理NaN值和数据类型，确保标识符字段是字符串
                cleaned_row = {}
                for key, value in row_dict.items():
                    if isinstance(value, np.ndarray):
                        cleaned_row[key] = value.tolist()
                    elif pd.isna(value):
                        cleaned_row[key] = None
                    elif isinstance(value, (np.integer, int)):
                        # 对于id和human_readable_id字段，必须是字符串
                        if key in ['id', 'human_readable_id']:
                            cleaned_row[key] = str(value)
                        else:
                            cleaned_row[key] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        cleaned_row[key] = float(value) if not pd.isna(value) else None
                    else:
                        cleaned_row[key] = value
                # 使用TextUnit的from_dict方法来正确构建对象
                try:
                    text_unit = TextUnit.from_dict(cleaned_row)
                except Exception:
                    # 如果from_dict失败，尝试手动构造
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

            # 尝试创建本地搜索引擎
            # 注意：这需要向量存储，而不仅仅是DataFrame
            from graphrag.vector_stores.lancedb import LanceDBVectorStore
            from graphrag.vector_stores.base import BaseVectorStore

            # 实际的向量存储路径
            entity_description_vector_store_path = lancedb_path / "default-entity-description.lance"
            community_full_content_vector_store_path = lancedb_path / "default-community-full_content.lance"
            text_unit_text_vector_store_path = lancedb_path / "default-text_unit-text.lance"

            # 检查向量存储是否存在
            if entity_description_vector_store_path.exists():
                # 创建向量存储配置
                # index_name: 向量数据库中集合的名称
                # id_field: 存储文档ID的字段名
                # text_field: 存储文本内容的字段名
                # vector_field: 存储向量嵌入的字段名
                # attributes_field: 存储额外属性的字段名
                # vector_size: 向量的维度（如text-embedding-ada-002是1536维）
                from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
                schema_config = VectorStoreSchemaConfig(
                    index_name="default-entity-description",
                    id_field="id",
                    text_field="text",
                    vector_field="vector",
                    attributes_field="attributes",
                    vector_size=1536
                )

                # 初始化向量存储
                entity_description_embedding_store = LanceDBVectorStore(
                    vector_store_schema_config=schema_config
                )
                # 连接数据库
                entity_description_embedding_store.connect(
                    db_uri=str(lancedb_path),
                    collection_name="default-entity-description"
                )
            else:
                # 如果向量存储不存在，返回错误
                return f"错误：未找到实体描述向量存储: {entity_description_vector_store_path}"

            # 创建本地搜索引擎
            # response_type选项:
            # - "general": 详细格式，包含数据引用标记
            # - "multiple paragraphs": 多段落格式，较少引用标记
            # - "single paragraph": 单段落格式，最简洁
            # - "list": 列表格式
            # print("创建本地搜索引擎...")
            search_engine = get_local_search_engine(
                config=config,
                reports=reports,
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                covariates={},
                response_type="general",  # 改为更简洁的输出格式
                description_embedding_store=entity_description_embedding_store
            )

            # # 执行查询
            # result = search_engine.search(query=query)
            # return str(result)

            # 执行查询 - search方法是异步的，需要在同步函数中调用
            # 使用 asyncio.run() 是在同步环境中运行异步代码的标准做法
            import asyncio

            try:
                if asyncio.iscoroutinefunction(search_engine.search):
                    # 如果search是协程函数，使用asyncio.run来运行它
                    # import pdb
                    # pdb.set_trace()
                    result = asyncio.run(search_engine.search(query=query))

                    # # --- 添加的调试代码 ---
                    # print(f"Type of result: {type(result)}")
                    # print(f"Content of result: {result}")

                    # # 如果 result 是一个对象，尝试查看它的所有属性
                    # if hasattr(result, '__dict__'):
                    #     print(f"Result attributes (vars): {vars(result)}")
                    # else:
                    #     # 如果没有 __dict__，用 dir() 查看所有方法和属性
                    #     print(f"Result attributes (dir): {[attr for attr in dir(result) if not attr.startswith('_')]}")
                    # # --- 调试代码结束 ---

                else:
                    # 如果不是（未来版本可能变化），则直接调用
                    result = search_engine.search(query=query)


                # 最终输出: 检查 result 对象是否有 'response' 属性
                if hasattr(result, 'response'):
                    # 如果有，返回答案
                    return result.response
                else:
                    # 如果没有，返回一个错误信息或者整个对象的字符串表示作为备选
                    self.logger.warning("Search result object does not have a 'response' attribute.")
                    return "错误：无法从搜索结果中提取答案。"
                # return str(result)
            except Exception as e:
                # 捕获 asyncio.run 或 search 本身可能抛出的异常
                self.logger.error(f"执行异步搜索时出错: {str(e)}")
                return f"执行异步搜索时出错: {str(e)}"

        except Exception as e:
            self.logger.error(f"本地搜索执行错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"本地搜索错误: {str(e)}"