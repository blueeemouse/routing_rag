"""
测试 Decomposer 使用 Ollama
Test Decomposer with Ollama
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decomposer.decomposer import Decomposer


def test_decomposer_with_ollama():
    """测试 Decomposer 使用 Ollama"""
    print("=" * 60)
    print("测试 Decomposer 使用 Ollama")
    print("=" * 60)

    try:
        # 初始化 Decomposer（使用 settings.yaml 中的配置）
        print("\n初始化 Decomposer...")
        decomposer = Decomposer()
        print("✓ Decomposer 初始化成功")

        # 测试查询分解
        test_queries = [
            "什么是人工智能？它有哪些应用？",
            "北京是中国的首都吗？",
            "1+1等于几？",
            "你好",
            "提出了相对论的科学家的最大的成就是？"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n测试查询 {i}: {query}")
            sub_queries = decomposer.decompose(query)
            print(f"分解结果 ({len(sub_queries)} 个子查询):")
            for j, sub_q in enumerate(sub_queries, 1):
                print(f"  {j}. {sub_q}")
            assert sub_queries is not None, "结果不能为 None"
            assert len(sub_queries) > 0, "结果不能为空"
            print(f"✓ 查询 {i} 测试通过")

        print("\n" + "=" * 60)
        print("✓ 所有 Decomposer Ollama 测试通过！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_decomposer_with_ollama()
    sys.exit(0 if success else 1)
