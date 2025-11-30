import yaml
import os
from typing import Dict, Any


class Config:
    """
    配置管理类
    Configuration Management Class
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        初始化配置管理器
        Initialize configuration manager

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        Load configuration from YAML file
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @property
    def decomposer_api_url(self) -> str:
        """获取decomposer API URL"""
        return self.config.get('decomposer', {}).get('api_url', '')

    @property
    def decomposer_api_key(self) -> str:
        """获取decomposer API密钥"""
        return self.config.get('decomposer', {}).get('api_key', '')

    @property
    def decomposer_model(self) -> str:
        """获取decomposer模型名称"""
        return self.config.get('decomposer', {}).get('model', 'gpt-3.5-turbo')

    @property
    def decomposer_prompt(self) -> str:
        """获取decomposer提示词模板"""
        return self.config.get('decomposer', {}).get('prompt',
            "将复杂查询分解为简单子查询。输入：{query}\n输出子查询，每行一个：")

    @property
    def router_api_url(self) -> str:
        """获取router API URL"""
        return self.config.get('router', {}).get('api_url', '')

    @property
    def router_api_key(self) -> str:
        """获取router API密钥"""
        return self.config.get('router', {}).get('api_key', '')

    @property
    def router_model(self) -> str:
        """获取router模型名称"""
        return self.config.get('router', {}).get('model', 'gpt-3.5-turbo')

    @property
    def router_prompt(self) -> str:
        """获取router提示词模板"""
        return self.config.get('router', {}).get('prompt',
            "确定查询处理策略：no_rag, naive_rag, 或 graph_rag。查询：{sub_query}\n策略：")

    @property
    def naive_rag_api_url(self) -> str:
        """获取naive_rag API URL"""
        return self.config.get('naive_rag', {}).get('api_url', '')

    @property
    def naive_rag_api_key(self) -> str:
        """获取naive_rag API密钥"""
        return self.config.get('naive_rag', {}).get('api_key', '')

    @property
    def naive_rag_model(self) -> str:
        """获取naive_rag模型名称"""
        return self.config.get('naive_rag', {}).get('model', 'gpt-3.5-turbo')

    @property
    def naive_rag_embedding_model(self) -> str:
        """获取naive_rag嵌入模型名称"""
        return self.config.get('naive_rag', {}).get('embedding_model', 'text-embedding-ada-002')

    @property
    def naive_rag_chunk_size(self) -> int:
        """获取naive_rag文档切块大小"""
        return self.config.get('naive_rag', {}).get('chunk_size', 512)

    @property
    def naive_rag_top_k(self) -> int:
        """获取naive_rag检索top-k值"""
        return self.config.get('naive_rag', {}).get('top_k', 5)

    @property
    def naive_rag_temperature(self) -> float:
        """获取naive_rag生成温度"""
        return self.config.get('naive_rag', {}).get('temperature', 0.7)

    @property
    def graph_rag_api_url(self) -> str:
        """获取graph_rag API URL"""
        return self.config.get('graph_rag', {}).get('api_url', '')

    @property
    def graph_rag_api_key(self) -> str:
        """获取graph_rag API密钥"""
        return self.config.get('graph_rag', {}).get('api_key', '')

    @property
    def graph_rag_model(self) -> str:
        """获取graph_rag模型名称"""
        return self.config.get('graph_rag', {}).get('model', 'gpt-3.5-turbo')

    @property
    def graph_rag_embedding_model(self) -> str:
        """获取graph_rag嵌入模型名称"""
        return self.config.get('graph_rag', {}).get('embedding_model', 'text-embedding-ada-002')

    @property
    def graph_rag_chunk_size(self) -> int:
        """获取graph_rag文档切块大小"""
        return self.config.get('graph_rag', {}).get('chunk_size', 512)

    @property
    def graph_rag_top_k(self) -> int:
        """获取graph_rag检索top-k值"""
        return self.config.get('graph_rag', {}).get('top_k', 5)

    @property
    def graph_rag_temperature(self) -> float:
        """获取graph_rag生成温度"""
        return self.config.get('graph_rag', {}).get('temperature', 0.7)

    @property
    def naive_rag_enabled(self) -> bool:
        """检查是否启用naive RAG"""
        return self.config.get('rag', {}).get('naive_rag_enabled', True)

    @property
    def graph_rag_enabled(self) -> bool:
        """检查是否启用graph RAG"""
        return self.config.get('rag', {}).get('graph_rag_enabled', True)


# 全局配置实例
settings = Config()