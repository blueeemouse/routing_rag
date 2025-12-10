import requests
from typing import Dict, Any
from interfaces.router_interface import RouterInterface
from config.config import settings


class Router(RouterInterface):
    def __init__(self):
        self.api_url = settings.router_api_url
        self.api_key = settings.router_api_key
        self.prompt_template = settings.router_prompt
        self.model = settings.router_model

    def route(self, sub_query: str) -> str:
        """
        确定如何处理子查询
        Determine how to process the sub-query

        Args:
            sub_query (str): 子查询字符串

        Returns:
            str: 处理策略（如 'no_rag', 'naive_rag', 'graph_rag'）
        """
        prompt = self.prompt_template.format(sub_query=sub_query)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0.1
        }

        try:
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()  # 检查 HTTP 错误
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', "").strip().lower()
                if 'no_rag' in content or 'no rag' in content:
                    return 'no_rag'
                elif 'naive_rag' in content or 'naive rag' in content:
                    return 'naive_rag'
                elif 'graph_rag' in content or 'graph rag' in content:
                    return 'graph_rag'
                else:
                    return 'no_rag'
            except Exception as e:
                # 捕获所有异常（如网络错误、API 错误等），并返回默认的 no_rag
                print(f"Error: {e}")  # 可选：打印错误日志
                return 'no_rag'

        except requests.exceptions.RequestException as e:
            print(f"Error routing query: {e}")
            return 'no_rag'