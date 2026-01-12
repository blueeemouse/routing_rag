"""
Test cases for the Decomposer module.

This test file verifies the functionality of the Decomposer class.
It uses mocking to simulate API responses without making actual API calls.
The tests ensure that the decomposer correctly handles query decomposition
and processes API responses according to the expected format.

Note: These tests use mock API responses and do not require valid API keys.
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


class TestDecomposer(unittest.TestCase):
    """
    Test class for the Decomposer module.
    Tests query decomposition functionality with mocked API responses.
    """

    def setUp(self):
        """
        Set up the test case with a Decomposer instance.
        The Decomposer will use the configuration from config/settings.yaml
        but API calls are mocked during testing.
        """
        self.decomposer = Decomposer()

    @patch('decomposer.decomposer.requests.post')
    def test_decompose_query(self, mock_post):
        """
        Test query decomposition with a mock API response.
        Verifies that the decomposer correctly processes the API response
        and splits the query into sub-queries.
        """
        # 模拟API响应，符合新实现的结构
        # Mock the API response to simulate a successful query decomposition
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "子查询1\n子查询2\n子查询3"
                    }
                }
            ]
        }
        result = self.decomposer.decompose("复杂查询示例")

        # 验证结果长度和内容
        # Verify the number of sub-queries and their content
        self.assertEqual(len(result), 3)
        self.assertIn("子查询1", result)
        self.assertIn("子查询2", result)
        self.assertIn("子查询3", result)

    # 这里在decomposer.decomposer模块把requests.post替换成下面的mock_post
    # 也就是用我们的模拟的结果代替真实的结果。主要是为了测试指定结果下的行为
    # 这里想测试的就是，当decomposer调用的api返回空结果的时候，decomposer理应返回原查询
    @patch('decomposer.decomposer.requests.post')
    def test_decompose_empty_response(self, mock_post):
        """
        Test decomposition with an empty API response.
        Verifies that the decomposer handles empty responses gracefully.
        """
        # 模拟空API响应
        # Mock an empty API response to test edge case handling
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": ""
                    }
                }
            ]
        }
        result = self.decomposer.decompose("复杂查询示例")

        # 验证返回原查询
        # Verify that an empty response results in an empty list
        # self.assertEqual(result, [])
        self.assertEqual(result, ["复杂查询示例"])



if __name__ == "__main__":
    """
    Main entry point for running the tests.
    Execute this file directly to run all test cases.
    """
    unittest.main()