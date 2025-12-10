"""
使用模拟组件测试orchestrator
Testing orchestrator with mock components
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import Orchestrator
from mocks.mock_components import MockDecomposer, MockRouter, MockNaiveRAG, MockGraphRAG, MockNoRAG


def test_orchestrator_with_mock_components():
    """
    使用模拟组件测试orchestrator
    """
    print("开始测试orchestrator与模拟组件...")
    
    # 创建模拟组件实例
    mock_decomposer = MockDecomposer()
    mock_router = MockRouter()
    mock_rag_implementations = {
        'naive_rag': MockNaiveRAG(),
        'graph_rag': MockGraphRAG(),
        'no_rag': MockNoRAG()
    }
    
    # 创建orchestrator实例
    orchestrator = Orchestrator(
        decomposer=mock_decomposer,
        router=mock_router,
        rag_implementations=mock_rag_implementations
    )
    
    # 测试用例1：复杂查询
    print("\n--- 测试用例1：复杂查询 ---")
    test_query_1 = "谁是美国总统且美国首都在哪里？"
    result_1 = orchestrator.process_query(test_query_1)
    print(f"最终结果:\n{result_1}")
    
    # 测试用例2：包含不同策略的查询
    print("\n--- 测试用例2：包含不同策略的查询 ---")
    test_query_2 = "北京的天气如何以及上海的GDP是多少？"
    result_2 = orchestrator.process_query(test_query_2)
    print(f"最终结果:\n{result_2}")
    
    # 测试用例3：简单查询
    print("\n--- 测试用例3：简单查询 ---")
    test_query_3 = "如何制作蛋糕"
    result_3 = orchestrator.process_query(test_query_3)
    print(f"最终结果:\n{result_3}")
    
    print("\n--- 测试完成 ---")


if __name__ == "__main__":
    test_orchestrator_with_mock_components()