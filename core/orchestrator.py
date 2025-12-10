"""
核心编排器模块
Core Orchestrator Module

主管道，结合所有组件
处理流程：输入查询 → 分解 → 路由 → 执行 → 合并结果
Main pipeline that combines all components
Handles the flow: Input Query → Decompose → Route → Execute → Combine Results
"""
from typing import Dict, Any, List
from interfaces.decomposer_interface import DecomposerInterface
from interfaces.router_interface import RouterInterface
from interfaces.rag_interface import RAGInterface


class Orchestrator:
    """
    RAG查询路由和分解框架的核心编排器
    Core orchestrator for the RAG query routing and decomposition framework
    """

    def __init__(self,
                 decomposer: DecomposerInterface,
                 router: RouterInterface,
                 rag_implementations: Dict[str, RAGInterface]):
        """
        初始化编排器

        Args:
            decomposer: 查询分解器
            router: 查询路由器
            rag_implementations: RAG实现映射，键为策略名称，值为对应的RAG实现
        """
        self.decomposer = decomposer
        self.router = router
        self.rag_implementations = rag_implementations

    def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        处理输入查询的主流程

        Args:
            query: 输入查询字符串
            context: 上下文信息（可选）

        Returns:
            str: 最终结果
        """
        print(f"开始处理查询: {query}")

        # 1. 分解查询
        subqueries = self.decompose_query(query)
        print(f"查询分解结果: {subqueries}")

        # 如果查询无法分解（返回空列表），将原始查询作为子查询
        if not subqueries:
            print(f"查询未被分解，使用原始查询: {query}")
            subqueries = [query]

        # 2. 路由子查询
        routed_queries = self.route_subqueries(subqueries)
        print(f"子查询路由结果: {routed_queries}")

        # 3. 执行子查询
        results = self.execute_subqueries(routed_queries, context)
        print(f"子查询执行结果: {results}")

        # 4. 合并结果
        final_result = self.combine_results(results)
        print(f"最终结果: {final_result}")

        return final_result

    def decompose_query(self, query: str) -> List[str]:
        """
        分解查询为子查询

        Args:
            query: 输入查询字符串

        Returns:
            List[str]: 子查询列表
        """
        return self.decomposer.decompose(query)

    def route_subqueries(self, subqueries: List[str]) -> List[Dict[str, Any]]:
        """
        为子查询确定路由策略

        Args:
            subqueries: 子查询列表

        Returns:
            List[Dict[str, Any]]: 包含子查询和对应路由策略的字典列表
        """
        routed_queries = []
        for subquery in subqueries:
            strategy = self.router.route(subquery)
            routed_queries.append({
                'query': subquery,
                'strategy': strategy
            })
        return routed_queries

    def execute_subqueries(self, routed_queries: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        执行已路由的子查询

        Args:
            routed_queries: 已路由的查询列表
            context: 上下文信息（可选）

        Returns:
            List[Dict[str, Any]]: 执行结果列表
        """
        results = []
        for item in routed_queries:
            query = item['query']
            strategy = item['strategy']

            # 获取对应的RAG实现
            rag_impl = self.rag_implementations.get(strategy)
            if rag_impl is None:
                result = f"错误：未找到策略 '{strategy}' 对应的RAG实现"
            else:
                # 执行查询，传递上下文（RAG实现可以选择是否使用）
                result = rag_impl.execute(query, context)

            results.append({
                'query': query,
                'strategy': strategy,
                'result': result
            })

        return results

    def combine_results(self, results: List[Dict[str, Any]]) -> str:
        """
        合并子查询执行结果

        Args:
            results: 子查询执行结果列表

        Returns:
            str: 合并后的最终结果
        """
        if not results:
            return "未生成任何结果"

        # 简单的合并策略：将所有结果连接起来
        combined_parts = []
        for item in results:
            combined_parts.append(f"查询: {item['query']}\n策略: {item['strategy']}\n结果: {item['result']}\n")

        return "\n---\n".join(combined_parts)