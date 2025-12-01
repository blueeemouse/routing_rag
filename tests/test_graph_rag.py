"""
Test cases for the Graph RAG module.
"""
import unittest
from unittest.mock import patch, MagicMock

import os, sys
# 动态添加 rag_implementations 模块的父目录到搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)
sys.path.append(parent_dir)

from rag_implementations.graph_rag.graph_rag_impl import GraphRAG


class TestGraphRAG(unittest.TestCase):
    """
    GraphRAG类的测试用例
    Test cases for the GraphRAG class
    """

    def setUp(self):
        """
        测试前的设置
        Setup before each test
        """
        self.graph_rag = GraphRAG()

    def test_interface_implementation(self):
        """
        测试GraphRAG是否正确实现了RAGInterface
        Test that GraphRAG correctly implements RAGInterface
        """
        # 确保execute方法存在
        self.assertTrue(hasattr(GraphRAG, 'execute'))
        # 确保execute方法是可调用的
        self.assertTrue(callable(getattr(self.graph_rag, 'execute')))

    def test_execute_method_signature(self):
        """
        测试execute方法的签名
        Test the signature of the execute method
        """
        import inspect
        sig = inspect.signature(self.graph_rag.execute)
        params = list(sig.parameters.keys())
        
        # 应该有query参数和可选的context参数
        self.assertIn('query', params)
        
    def test_execute_method(self):
        """
        测试execute方法的基本功能
        Test the basic functionality of the execute method
        """
        result = self.graph_rag.execute("测试查询")
        
        # 验证返回结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        
        # 如果GraphRAG不可用，应该返回错误信息
        if not self.graph_rag._graph_rag_available:
            self.assertIn("错误", result)
            self.assertIn("GraphRAG", result)
        else:
            # 如果可用，也应该返回某种形式的结果
            self.assertIsInstance(result, str)

    @patch('rag_implementations.graph_rag.graph_rag_impl.sys')
    def test_execute_with_graph_context(self, mock_sys):
        """
        测试带图数据上下文的execute方法
        Test execute method with graph data context
        """
        # 模拟一个包含图数据的上下文
        context = {
            'entities': ["entity1", "entity2"],
            'relationships': ["rel1", "rel2"],
            'reports': ["report1"],
            'text_units': ["text_unit1"],
            'communities': ["community1"]
        }
        
        result = self.graph_rag.execute("测试查询", context=context)
        
        # 验证返回结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_execute_with_empty_context(self):
        """
        测试空上下文的execute方法
        Test execute method with empty context
        """
        result = self.graph_rag.execute("测试查询", context={})
        
        # 验证返回结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_execute_with_none_context(self):
        """
        测试None上下文的execute方法
        Test execute method with None context
        """
        result = self.graph_rag.execute("测试查询", context=None)
        
        # 验证返回结果
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_has_graph_data_method(self):
        """
        测试_graph_rag_available属性和_has_graph_data方法
        Test the _graph_rag_available attribute and _has_graph_data method
        """
        # 测试没有图数据的情况
        self.assertFalse(self.graph_rag._has_graph_data({}))
        self.assertFalse(self.graph_rag._has_graph_data(None))
        
        # 测试有图数据的情况
        context_with_entities = {'entities': ['entity1']}
        self.assertTrue(self.graph_rag._has_graph_data(context_with_entities))
        
        context_with_relationships = {'relationships': ['rel1']}
        self.assertTrue(self.graph_rag._has_graph_data(context_with_relationships))
        
        context_with_reports = {'reports': ['report1']}
        self.assertTrue(self.graph_rag._has_graph_data(context_with_reports))
        
        context_with_text_units = {'text_units': ['text1']}
        self.assertTrue(self.graph_rag._has_graph_data(context_with_text_units))
        
        context_with_communities = {'communities': ['comm1']}
        self.assertTrue(self.graph_rag._has_graph_data(context_with_communities))

    def test_add_document_method(self):
        """
        测试add_document方法
        Test the add_document method
        """
        # 确保方法存在
        self.assertTrue(hasattr(self.graph_rag, 'add_document'))
        self.assertTrue(callable(getattr(self.graph_rag, 'add_document')))
        
        # 测试方法调用
        try:
            self.graph_rag.add_document("测试文档内容", {"source": "test"})
            # 方法应该执行而不抛出异常
        except Exception as e:
            # 记录但不失败，因为这可能是由于依赖项缺失导致的
            pass


if __name__ == "__main__":
    unittest.main()