"""
Test cases for the Router module.

This test file verifies the functionality of the Router class.
It uses mocking to simulate API responses without making actual API calls.
The tests ensure that the router correctly handles query routing
and processes API responses to determine appropriate strategies.

Strategies:
- no_rag: No RAG processing, direct response
- naive_rag: Simple RAG processing
- graph_rag: Graph-based RAG processing

Note: These tests use mock API responses and do not require valid API keys.
"""

import os, sys
# 动态添加 rag_implementations 模块的父目录到搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)
sys.path.append(parent_dir)


import unittest
from unittest.mock import patch
from router.router import Router


class TestRouter(unittest.TestCase):
    """
    Test class for the Router module.
    Tests query routing functionality with mocked API responses.
    Verifies that the router correctly identifies and returns
    appropriate processing strategies for different query types.
    """

    def setUp(self):
        """
        Set up the test case with a Router instance.
        The Router will use the configuration from config/settings.yaml
        but API calls are mocked during testing.
        """
        self.router = Router()

    @patch('router.router.requests.post')
    def test_route_to_no_rag(self, mock_post):
        """
        Test routing to no_rag strategy.
        Verifies that simple queries are correctly routed to no_rag strategy.
        """
        # 模拟API响应，返回no_rag策略
        # Mock API response for no_rag strategy
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "no_rag"
                    }
                }
            ]
        }
        result = self.router.route("简单查询示例")
        self.assertEqual(result, 'no_rag')

    @patch('router.router.requests.post')
    def test_route_to_naive_rag(self, mock_post):
        """
        Test routing to naive_rag strategy.
        Verifies that knowledge-intensive queries are routed to naive_rag strategy.
        """
        # 模拟API响应，返回naive_rag策略
        # Mock API response for naive_rag strategy
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "naive_rag"
                    }
                }
            ]
        }
        result = self.router.route("需要检索的查询示例")
        self.assertEqual(result, 'naive_rag')

    @patch('router.router.requests.post')
    def test_route_to_graph_rag(self, mock_post):
        """
        Test routing to graph_rag strategy.
        Verifies that relationship-intensive queries are routed to graph_rag strategy.
        """
        # 模拟API响应，返回graph_rag策略
        # Mock API response for graph_rag strategy
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "graph_rag"
                    }
                }
            ]
        }
        result = self.router.route("复杂关系查询示例")
        self.assertEqual(result, 'graph_rag')

    @patch('router.router.requests.post')
    def test_route_with_api_error(self, mock_post):
        """
        Test routing with API error fallback.
        Verifies that the router returns a default strategy when API calls fail.
        """
        # 模拟API错误情况
        # Mock an API error to test fallback behavior
        mock_post.side_effect = Exception("API Error")
        result = self.router.route("出错查询示例")
        # 应该返回默认值 no_rag
        # Should return default strategy on error
        self.assertEqual(result, 'no_rag')

    @patch('router.router.requests.post')
    def test_route_with_unrecognized_response(self, mock_post):
        """
        Test routing with unrecognized response.
        Verifies that the router handles unrecognized API responses gracefully.
        """
        # 模拟API返回无法识别的响应
        # Mock an unrecognized API response to test handling
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "unknown_strategy"
                    }
                }
            ]
        }
        result = self.router.route("未知策略查询示例")
        # 应该返回默认值 no_rag
        # Should return default strategy for unrecognized responses
        self.assertEqual(result, 'no_rag')


if __name__ == "__main__":
    """
    Main entry point for running the tests.
    Execute this file directly to run all test cases.
    """
    unittest.main()