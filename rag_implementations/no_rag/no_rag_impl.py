"""
No RAG Implementation - 直接调用LLM回答，无检索增强
Direct LLM response without retrieval augmentation
"""
import requests
from typing import Dict, Any
from interfaces.rag_interface import RAGInterface
from config.config import settings
import logging
import time


class NoRAG(RAGInterface):
    """
    无RAG实现 - 直接调用LLM模型回答问题
    No RAG implementation - Directly call LLM to answer questions
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化NoRAG

        Args:
            config: 配置参数字典
        """
        self.config = config or {}

        # 优先使用传入的配置，否则使用全局配置
        self.api_url = self.config.get('api_url', settings.naive_rag_api_url)
        # print(f"Using NoRAG api_url: {self.api_url}")
        # 智能补全后缀
        if not self.api_url.endswith("/chat/completions"):
            self.api_url = self.api_url.rstrip('/') + "/chat/completions"

        self.api_key = self.config.get('api_key', settings.naive_rag_api_key)
        self.model = self.config.get('model', settings.naive_rag_model)
        self.temperature = self.config.get('temperature', settings.naive_rag_temperature)

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 记录最后一次生成的时间和token信息
        self.last_generation_time = 0.0
        self.last_generation_tokens = 0
        self.last_total_tokens = 0

    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        直接调用LLM回答查询，不使用检索增强
        Directly call LLM to answer query without retrieval augmentation

        Args:
            query (str): 查询字符串
            context (Dict[str, Any], optional): 上下文信息

        Returns:
            str: LLM生成的回答
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 200,
            "temperature": self.temperature
        }
        
        try:
            # 记录开始时间
            generation_start = time.time()

            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            # 记录结束时间
            generation_end = time.time()

            # 提取AI响应内容
            content = result.get('choices', [{}])[0].get('message', {}).get('content', "")

            # 提取usage信息
            usage = result.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

            # 记录生成时间和token信息
            self.last_generation_time = generation_end - generation_start
            self.last_generation_tokens = completion_tokens
            self.last_total_tokens = total_tokens

            return content.strip()
        except Exception as e:
            self.logger.error(f"NoRAG调用LLM时出错: {str(e)}")
            return f"直接回答: {query}"

    def build_index_from_data(self, data, metadata=None, **kwargs):
        """
        NoRAG不需要构建索引
        NoRAG doesn't need to build index
        """
        # NoRAG不需要索引，直接返回成功
        return True

    def build_index_from_path(self, root_dir: str, config_filepath: str = None, output_dir: str = None, **kwargs):
        """
        NoRAG不需要构建索引
        NoRAG doesn't need to build index
        """
        # NoRAG不需要索引，直接返回成功
        return True