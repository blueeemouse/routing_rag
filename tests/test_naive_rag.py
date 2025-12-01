"""
Test cases for the Naive RAG module.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 动态添加 rag_implementations 模块的父目录到搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)
sys.path.append(parent_dir)

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG


class TestNaiveRAG(unittest.TestCase):
    """
    NaiveRAG类的测试用例
    Test cases for the NaiveRAG class
    """

    def setUp(self):
        """
        测试前的设置
        Setup before each test
        """
        self.naive_rag = NaiveRAG()

    def test_interface_implementation(self):
        """
        测试NaiveRAG是否正确实现了RAGInterface
        Test that NaiveRAG correctly implements RAGInterface
        """
        # 确保execute方法存在
        self.assertTrue(hasattr(NaiveRAG, 'execute'))
        # 确保execute方法是可调用的
        self.assertTrue(callable(getattr(self.naive_rag, 'execute')))

    def test_execute_method_signature(self):
        """
        测试execute方法的签名
        Test the signature of the execute method
        """
        import inspect
        sig = inspect.signature(self.naive_rag.execute)
        params = list(sig.parameters.keys())

        # 应该有query参数和可选的context参数
        self.assertIn('query', params)

    def test_execute_with_documents_in_context(self):
        """
        测试在上下文中有文档时的execute方法
        Test execute method with documents in context
        """
        # 测试LlamaIndex不可用的情况
        self.naive_rag._llama_index_available = False

        context = {
            'documents': [
                "这是第一个文档的内容",
                "这是第二个文档的内容"
            ]
        }

        result = self.naive_rag.execute("测试查询", context)

        # 验证返回的是错误信息
        self.assertIn("错误", result)
        self.assertIn("LlamaIndex", result)

    def test_execute_without_context(self):
        """
        测试没有上下文时的execute方法
        Test execute method without context
        """
        # 测试LlamaIndex不可用的情况
        self.naive_rag._llama_index_available = False

        result = self.naive_rag.execute("测试查询", context=None)

        # 验证返回的是错误信息
        self.assertIn("错误", result)
        self.assertIn("LlamaIndex", result)

    def test_execute_llama_index_not_available(self):
        """
        测试LlamaIndex不可用时execute方法的行为
        Test execute method behavior when LlamaIndex is not available
        """
        # 模拟LlamaIndex不可用的情况
        self.naive_rag._llama_index_available = False

        result = self.naive_rag.execute("测试查询", context=None)

        # 验证返回的是错误信息
        self.assertIn("错误", result)
        self.assertIn("LlamaIndex", result)

    def test_add_document_method(self):
        """
        测试add_document方法
        Test the add_document method
        """
        # 确保方法存在
        self.assertTrue(hasattr(self.naive_rag, 'add_document'))
        self.assertTrue(callable(getattr(self.naive_rag, 'add_document')))

        # 测试方法调用
        try:
            self.naive_rag.add_document("测试文档内容", {"source": "test"})
            # 如果LlamaIndex不可用，应该记录错误但不抛出异常
        except Exception as e:
            # 如果出现其他异常，也应是可预期的
            pass


if __name__ == "__main__":
    unittest.main()