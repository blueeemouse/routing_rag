"""
测试API调用的验证脚本，现在包括RAG实现
用于测试decomposer、router以及naive RAG和graph RAG的基本功能
"""

import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()

parent_dir_path = current_file_path.parent.parent

sys.path.append(str(parent_dir_path))

from decomposer.decomposer import Decomposer
from router.router import Router
from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
import os


def test_decomposer_api():
    """
    测试decomposer的API调用
    """
    print("=" * 50)
    print("测试 Decomposer API 调用")
    print("=" * 50)

    try:
        decomposer = Decomposer()

        # 测试查询分解
        test_query = "分析2023年人工智能发展状况，包括技术突破、行业应用和未来趋势"
        # test_query = "分析我国2024年经济发展前景，包括主要驱动力和潜在挑战"
        print(f"原始查询: {test_query}")

        sub_queries = decomposer.decompose(test_query)
        print(f"分解结果: {sub_queries}")
        print(f"分解出 {len(sub_queries)} 个子查询")

        # 测试简单查询
        simple_query = "今天天气如何"
        print(f"\n原始查询: {simple_query}")

        simple_sub_queries = decomposer.decompose(simple_query)
        print(f"分解结果: {simple_sub_queries}")
        print(f"分解出 {len(simple_sub_queries)} 个子查询")

        print("\n[完成] Decomposer API 调用测试完成")

    except Exception as e:
        print(f"\n❌ Decomposer API 调用失败: {str(e)}")
        print("请确保在 config/settings.yaml 中设置了正确的API密钥")


def test_router_api():
    """
    测试router的API调用
    """
    print("\n" + "=" * 50)
    print("测试 Router API 调用")
    print("=" * 50)

    try:
        router = Router()

        # 测试不同类型的查询路由
        queries_and_expected = [
            ("今天天气如何", "no_rag"),
            ("如何做番茄炒蛋", "no_rag"),
            ("2023年人工智能最新技术", "naive_rag"),
            ("美国独立战争对现代政治的影响", "naive_rag"),
            ("分析公司内部员工关系网络", "graph_rag")
        ]

        for query, expected_type in queries_and_expected:
            print(f"\n子查询: {query}")
            strategy = router.route(query)
            print(f"路由策略: {strategy}")

        print("\n[完成] Router API 调用测试完成")

    except Exception as e:
        print(f"\n[错误] Router API 调用失败: {str(e)}")
        print("请确保在 config/settings.yaml 中设置了正确的API密钥")


def test_naive_rag():
    """
    测试naive RAG的基本功能
    """
    print("\n" + "=" * 50)
    print("测试 Naive RAG 功能")
    print("=" * 50)

    try:
        naive_rag = NaiveRAG()

        print(f"API URL: {naive_rag.api_url}")
        print(f"模型: {naive_rag.model}")
        print(f"嵌入模型: {naive_rag.embedding_model}")

        # 测试execute方法
        result = naive_rag.execute("什么是人工智能？")
        # print(f"查询结果: {result}")

        # 如果LlamaIndex可用，将返回实际查询结果
        # 如果不可用，将返回错误信息
        if "错误" in result or "LlamaIndex" in result:
            print("[警告] LlamaIndex库不可用或配置有问题，这是预期的环境限制")
        else:
            print("[完成] Naive RAG 功能测试完成")

    except Exception as e:
        print(f"\n❌ Naive RAG 功能测试失败: {str(e)}")


def test_graph_rag():
    """
    测试graph RAG的基本功能
    """
    print("\n" + "=" * 50)
    print("测试 Graph RAG 功能")
    print("=" * 50)

    try:
        graph_rag = GraphRAG()

        print(f"API URL: {graph_rag.api_url}")
        print(f"模型: {graph_rag.model}")
        print(f"嵌入模型: {graph_rag.embedding_model}")

        # 测试execute方法
        # 注意路径可能会被转义导致错误.最好全都用双斜杠(如果是Linux系统或mac os,就不会有这个问题了.因为那边都是用正斜杠作为路径分隔符
        # 而正斜杠在python字符串里不是转义字符,因此不会有类似的问题.Windows用反斜杠就可能产生问题)
        # 或者用原始字符串 (r"...") 来表示路径(不过这个只能在Windows上用)
        # 最通用还是用pathlib里的 Path 来包装一下
        param = {"search_mode":'local', "data_path": "D:\Develop\\all_RAG\\routing_rag\graphrag_class_test_data"}
        # result = graph_rag.execute("分析公司内部员工关系", param)
        # # 在这里设置断点！
        # import pdb; pdb.set_trace() 
        result = graph_rag.execute("什么是人工智能？", param)
        print(f"查询结果: {result}")

        # 检查GraphRAG是否可用
        if "错误" in result or "GraphRAG" in result:
            print("[警告] GraphRAG库不可用或配置有问题，这是预期的环境限制")
        else:
            print("[完成] Graph RAG 功能测试完成")

    except Exception as e:
        print(f"\n❌ Graph RAG 功能测试失败: {str(e)}")


def main():
    """
    主函数，运行所有测试
    """
    print("开始测试API调用和RAG功能...")

    print("\n[警告] 注意：要成功运行此测试，您需要在 config/settings.yaml 中设置有效的API密钥")
    print("当前配置的API URL:", os.environ.get('LLM_API_URL', '请检查 config/settings.yaml'))

    # test_decomposer_api()
    # test_router_api()
    # 前面两个已经测试成功了，就是简单的调用api

    # test_naive_rag()  # 这个也通过了
    test_graph_rag()

    print("\n" + "=" * 50)
    print("所有API调用和RAG功能测试完成")
    print("=" * 50)
    print("如果出现API错误，请检查：")
    print("1. config/settings.yaml 中的API密钥是否正确")
    print("2. 网络连接是否正常")
    print("3. API服务商是否正常运行")
    print("\n对于RAG实现，如果提示LlamaIndex或GraphRAG不可用，请检查相应依赖是否安装")


if __name__ == "__main__":
    main()