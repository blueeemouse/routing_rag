import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()

parent_dir_path = current_file_path.parent.parent

sys.path.append(str(parent_dir_path))

import requests
from typing import List
from interfaces.decomposer_interface import DecomposerInterface
from config.config import settings


class Decomposer(DecomposerInterface):
    def __init__(self):
        self.api_url = settings.decomposer_api_url
        self.api_key = settings.decomposer_api_key
        self.prompt_template = settings.decomposer_prompt
        self.model = settings.decomposer_model

    def decompose(self, query: str) -> List[str]:
        """
        将复杂查询分解为子查询
        Split complex query into sub-queries

        Args:
            query: 要分解的输入查询

        Returns:
            子查询列表
        """
        prompt = self.prompt_template.format(query=query)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        # 提取AI响应内容
        content = result.get('choices', [{}])[0].get('message', {}).get('content', "")
        # 按行分割并过滤空行
        sub_queries = [q.strip() for q in content.split('\n') if q.strip()]
        return sub_queries