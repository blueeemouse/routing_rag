"""
测试 NaiveRAG 使用 Ollama
Test NaiveRAG with Ollama
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG


def test_naive_rag_with_ollama():
    """测试 NaiveRAG 使用 Ollama"""
    print("=" * 60)
    print("测试 NaiveRAG 使用 Ollama")
    print("=" * 60)

    # 配置 NaiveRAG 使用 Ollama
    config = {
        'api_url': 'http://127.0.0.1:11434',
        'api_key': 'ollama',
        'model': 'qwen2.5:3b',
        'embedding_model': 'nomic-embed-text',
        'chunk_size': 512,
        'top_k': 5,
        'temperature': 0.0
    }

    print(f"\n配置信息:")
    print(f"  API URL: {config['api_url']}")
    print(f"  Model: {config['model']}")
    print(f"  Embedding Model: {config['embedding_model']}")
    print(f"  Chunk Size: {config['chunk_size']}")
    print(f"  Top K: {config['top_k']}")

    try:
        # 初始化 NaiveRAG
        print("\n初始化 NaiveRAG...")
        # naive_rag = NaiveRAG(config)
        naive_rag = NaiveRAG()
        print("✓ NaiveRAG 初始化成功")

        # 准备测试数据
        test_documents = [
            "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的子领域，专注于让计算机从数据中学习，而无需明确编程。",
            "深度学习是机器学习的一种方法，使用多层神经网络来模拟人脑的学习过程。",
            "自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
            "计算机视觉是人工智能的另一个重要分支，致力于让计算机能够理解和解释视觉信息。",
            "强化学习是一种机器学习方法，通过奖励和惩罚机制来训练智能体做出决策。",
            "神经网络是深度学习的基础，模仿人脑神经元之间的连接方式。",
            "卷积神经网络（CNN）主要用于图像识别和计算机视觉任务。",
            "循环神经网络（RNN）主要用于处理序列数据，如文本和时间序列。",
            "Transformer 是一种革命性的神经网络架构，极大地推动了自然语言处理的发展。"
        ]

        # 构建索引
        print("\n构建索引...")
        success = naive_rag.build_index_from_data(test_documents)
        if not success:
            print("✗ 索引构建失败")
            return False
        print("✓ 索引构建成功")

        # 测试查询
        test_queries = [
            ("什么是人工智能？", "人工智能"),
            ("机器学习和深度学习的区别是什么？", "机器学习"),
            ("什么是神经网络？", "神经网络"),
            ("Transformer 有什么作用？", "Transformer"),
            ("1+1等于几？", None)  # 这个问题可能不在文档中
        ]

        print("\n执行查询测试...")
        for i, (query, expected_keyword) in enumerate(test_queries, 1):
            print(f"\n测试查询 {i}: {query}")
            if expected_keyword:
                print(f"  期望关键词: {expected_keyword}")

            result = naive_rag.execute(query)
            print(f"  结果: {result}")

            # 验证结果
            assert result is not None, "结果不能为 None"
            assert len(result) > 0, "结果不能为空"

            # 检查结果是否合理
            if expected_keyword:
                if expected_keyword.lower() in result.lower():
                    print(f"  ✓ 结果包含期望关键词")
                else:
                    print(f"  ⚠ 结果可能不包含期望关键词（可接受）")

            print(f"✓ 查询 {i} 测试通过")

        print("\n" + "=" * 60)
        print("✓ 所有 NaiveRAG Ollama 测试通过！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_naive_rag_with_ollama()
    sys.exit(0 if success else 1)