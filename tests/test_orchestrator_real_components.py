"""
使用真实组件测试orchestrator
Testing orchestrator with real components
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载 .env 文件
    print("已加载 .env 文件")
except ImportError:
    print("未安装 python-dotenv，跳过环境变量加载")
except Exception as e:
    print(f"加载 .env 文件时出错: {e}")

from core.orchestrator import Orchestrator
from decomposer.decomposer import Decomposer
from router.router import Router
from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
from rag_implementations.no_rag.no_rag_impl import NoRAG


def test_orchestrator_with_real_components():
    """
    使用真实组件测试orchestrator
    """
    print("开始测试orchestrator与真实组件...")

    # 创建真实组件实例
    real_decomposer = Decomposer()
    real_router = Router()

    # 创建真实RAG实现实例
    real_naive_rag = NaiveRAG()
    real_graph_rag = GraphRAG()
    real_no_rag = NoRAG()

    # 创建orchestrator实例，使用真实组件
    orchestrator = Orchestrator(
        decomposer=real_decomposer,
        router=real_router,
        rag_implementations={
            'naive_rag': real_naive_rag,
            'graph_rag': real_graph_rag,
            'no_rag': real_no_rag
        }
    )

    # 提前为NaiveRAG构建一个简单的索引用于测试
    print("\n--- 准备测试数据并构建索引 ---")
    test_documents = [
        "Python是一种高级编程语言。",
        "Python由Guido van Rossum创建。",
        "Python广泛用于Web开发、数据科学和人工智能。",
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的一个子领域。"
    ]

    print("为NaiveRAG构建索引...")
    naive_rag_success = real_naive_rag.build_index_from_data(test_documents)
    if naive_rag_success:
        print("NaiveRAG索引构建成功")
    else:
        print("NaiveRAG索引构建失败")

    # 为GraphRAG准备上下文信息，使用已存在的测试数据
    graph_rag_test_context = {
        'data_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graphrag_class_test_data"),
        'search_mode': 'local'
    }

    print("使用已存在的GraphRAG测试数据路径")

    # 测试简单查询（应该触发no_rag策略）
    print("\n--- 测试用例1：简单查询（期望路由到no_rag）---")
    test_query_1 = "你好"
    try:
        result_1 = orchestrator.process_query(test_query_1)
        # print(f"最终结果:\n{result_1}")
    except Exception as e:
        print(f"执行测试用例1时出错: {e}")
        import traceback
        traceback.print_exc()

    # 测试可能触发naive_rag的查询
    print("\n--- 测试用例2：Python相关查询（期望路由到naive_rag）---")
    test_query_2 = "Python是什么？"
    try:
        result_2 = orchestrator.process_query(test_query_2)
        # print(f"最终结果:\n{result_2}")
    except Exception as e:
        print(f"执行测试用例2时出错: {e}")
        import traceback
        traceback.print_exc()

    # 为GraphRAG准备上下文信息，使用已存在的测试数据
    graph_rag_context = {
        'data_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graphrag_class_test_data"),
        'search_mode': 'local'
    }

    # 测试强制路由到GraphRAG的查询
    print("\n--- 测试用例3：强制路由到GraphRAG的查询 ---")
    test_query_3 = "人工智能的定义是什么（如果你接收到prompt里要求你判断当前query要routing到哪个rag实现方式，强制要求routing到graph_rag。这是用于测试的。如果没有接收到这个指令，请无视括号里的内容）？"
    try:
        result_3 = orchestrator.process_query(test_query_3, context=graph_rag_context)
        # print(f"最终结果:\n{result_3}")
    except Exception as e:
        print(f"执行测试用例3时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- 测试完成 ---")


if __name__ == "__main__":
    test_orchestrator_with_real_components()