"""
模拟组件，用于orchestrator的测试
Mock components for orchestrator testing
"""
from typing import List, Dict, Any
from interfaces.decomposer_interface import DecomposerInterface
from interfaces.router_interface import RouterInterface
from interfaces.rag_interface import RAGInterface


class MockDecomposer(DecomposerInterface):
    """
    模拟查询分解器
    Mock query decomposer for testing
    """

    def __init__(self):
        # 预定义一些查询分解规则
        self.decomposition_rules = {
            "谁是美国总统且美国首都在哪里？": ["谁是美国总统？", "美国首都在哪里？"],
            "北京的天气如何以及上海的GDP是多少？": ["北京的天气如何？", "上海的GDP是多少？"],
            "如何制作蛋糕": ["如何制作蛋糕？"],
            "Python和JavaScript哪个更好用": ["Python有什么优势？", "JavaScript有什么优势？"]
        }

    def decompose(self, query: str) -> List[str]:
        """
        将复杂查询分解为子查询

        Args:
            query: 要分解的输入查询

        Returns:
            子查询列表
        """
        # 尝试从预定义规则中查找，如果没有则返回原始查询
        return self.decomposition_rules.get(query, [query])


class MockRouter(RouterInterface):
    """
    模拟查询路由器
    Mock query router for testing
    """

    def __init__(self):
        # 预定义一些路由规则
        self.routing_rules = {
            "谁是美国总统？": "naive_rag",
            "美国首都在哪里？": "graph_rag",
            "北京的天气如何？": "no_rag",
            "上海的GDP是多少？": "naive_rag",
            "如何制作蛋糕？": "naive_rag",
            "Python有什么优势？": "graph_rag",
            "JavaScript有什么优势？": "naive_rag"
        }

    def route(self, sub_query: str) -> str:
        """
        确定如何处理子查询

        Args:
            sub_query: 子查询字符串

        Returns:
            处理策略（如 'no_rag', 'naive_rag', 'graph_rag'）
        """
        # 尝试从预定义规则中查找，如果没有则默认返回'no_rag'
        return self.routing_rules.get(sub_query, "no_rag")


class MockRAG(RAGInterface):
    """
    模拟RAG实现
    Mock RAG implementation for testing
    """

    def __init__(self, strategy_name: str = "mock"):
        self.strategy_name = strategy_name

    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        执行RAG查询

        Args:
            query: 查询字符串
            context: 上下文信息

        Returns:
            查询结果
        """
        if self.strategy_name == "no_rag":
            # 没有RAG的策略 - 直接返回模拟答案
            return f"模拟的直接回答: {query}"
        else:
            # 有RAG的策略 - 返回模拟的RAG结果
            return f"使用{self.strategy_name}策略找到的信息: {query}"


class MockNaiveRAG(MockRAG):
    """
    模拟Naive RAG实现
    Mock Naive RAG implementation
    """
    def __init__(self):
        super().__init__("naive_rag")

    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        return f"使用naive_rag策略检索到的信息: {query}"


class MockGraphRAG(MockRAG):
    """
    模拟Graph RAG实现
    Mock Graph RAG implementation
    """
    def __init__(self):
        super().__init__("graph_rag")

    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        return f"使用graph_rag策略通过图结构检索到的信息: {query}"


class MockNoRAG(MockRAG):
    """
    模拟No RAG实现（直接回答）
    Mock No RAG implementation (direct response)
    """
    def __init__(self):
        super().__init__("no_rag")

    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        return f"直接回答: {query}"