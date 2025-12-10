"""
No RAG Implementation - 直接调用LLM回答，无检索增强
Direct LLM response without retrieval augmentation
"""
import requests
from typing import Dict, Any
from interfaces.rag_interface import RAGInterface
from config.config import settings
import logging


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
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
        # 下面是用于调试的代码段，可以帮助查看响应内容
        # try:
        #     response = requests.post(self.api_url, headers=headers, json=data)
            
        #     # --- 调试代码开始 ---
        #     print(f"状态码: {response.status_code}")
        #     print(f"响应头: {response.headers}")
        #     print(f"原始响应内容: {response.text}") # 这是关键！
        #     # --- 调试代码结束 ---

        #     response.raise_for_status() # 检查状态码
            
        #     result = response.json() # 在这里解析 JSON
            
            # 提取AI响应内容
            content = result.get('choices', [{}])[0].get('message', {}).get('content', "")
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