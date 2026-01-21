"""
测试 NoRAG 使用 Ollama
Test NoRAG with Ollama
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_implementations.no_rag.no_rag_impl import NoRAG


def test_norag_with_ollama():
    """测试 NoRAG 使用 Ollama"""
    print("=" * 60)
    print("测试 NoRAG 使用 Ollama")
    print("=" * 60)

    # 配置 NoRAG 使用 Ollama
    config = {
        'api_url': 'http://127.0.0.1:11434/v1/chat/completions',
        'api_key': 'ollama',
        'model': 'qwen2.5:3b',
        'temperature': 0.0
    }

    print(f"\n配置信息:")
    print(f"  API URL: {config['api_url']}")
    print(f"  Model: {config['model']}")
    print(f"  Temperature: {config['temperature']}")

    try:
        # 初始化 NoRAG
        print("\n初始化 NoRAG...")
        no_rag = NoRAG(config)
        print("✓ NoRAG 初始化成功")

        # 测试查询
        test_queries = [
            "什么是人工智能？",
            "1+1等于几？",
            "北京是中国的首都吗？"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n测试查询 {i}: {query}")
            result = no_rag.execute(query)
            print(f"结果: {result}")
            assert result is not None, "结果不能为 None"
            assert len(result) > 0, "结果不能为空"
            print(f"✓ 查询 {i} 测试通过")

        print("\n" + "=" * 60)
        print("✓ 所有 NoRAG Ollama 测试通过！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_norag_with_ollama()
    sys.exit(0 if success else 1)