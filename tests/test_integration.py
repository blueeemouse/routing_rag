"""
Test cases for the integrated workflow (Decomposer + Router).
"""
import unittest
from unittest.mock import patch

import os, sys
# 动态添加 rag_implementations 模块的父目录到搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
parent_dir = os.path.dirname(current_dir)
# print(parent_dir)
sys.path.append(parent_dir)

from decomposer.decomposer import Decomposer
from router.router import Router

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.decomposer = Decomposer()
        self.router = Router()

    @patch('decomposer.requests.post')
    @patch('router.requests.post')
    def test_full_workflow(self, mock_router_post, mock_decomposer_post):
        """Test the full workflow from query decomposition to routing."""
        mock_decomposer_post.return_value.json.return_value = {"sub_queries": ["query1", "query2"]}
        mock_router_post.return_value.json.return_value = {"method": "naive_rag"}

        sub_queries = self.decomposer.decompose("complex query")
        methods = [self.router.route(query) for query in sub_queries]

        self.assertEqual(len(sub_queries), 2)
        self.assertEqual(methods, ["naive_rag", "naive_rag"])

if __name__ == "__main__":
    unittest.main()