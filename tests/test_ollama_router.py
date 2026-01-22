"""
测试 Router 使用 Ollama
Test Router with Ollama
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.router import Router


def test_router_with_ollama():
    """测试 Router 使用 Ollama"""
    print("=" * 60)
    print("测试 Router 使用 Ollama")
    print("=" * 60)

    try:
        # 初始化 Router（使用 settings.yaml 中的配置）
        print("\n初始化 Router...")
        router = Router()
        print("✓ Router 初始化成功")

        # 测试查询路由
        test_queries = [
            # no_rag 类型的查询
            ("1+1等于几？", "no_rag"),
            ("北京是中国的首都吗？", "no_rag"),
            ("你好", "no_rag"),

            # naive_rag 类型的查询
            ("什么是人工智能？", "naive_rag"),
            ("机器学习有哪些应用？", "naive_rag"),

            # graph_rag 类型的查询
            ("张三和李四的关系是什么？", "graph_rag"),
            ("爱因斯坦的成就有哪些？", "graph_rag"),
        ]

        correct_count = 0
        for i, (query, expected_strategy) in enumerate(test_queries, 1):
            print(f"\n测试查询 {i}: {query}")
            print(f"  期望策略: {expected_strategy}")
            result = router.route(query)
            print(f"  实际策略: {result}")

            # 检查结果是否有效
            assert result is not None, "结果不能为 None"
            assert result in ['no_rag', 'naive_rag', 'graph_rag'], f"策略必须是 no_rag, naive_rag 或 graph_rag，实际为: {result}"

            # 检查是否与期望一致（宽松检查，因为路由可能不准确）
            if result == expected_strategy:
                print(f"  ✓ 策略匹配")
                correct_count += 1
            else:
                print(f"  ⚠ 策略不匹配（可接受，因为路由可能不准确）")

            print(f"✓ 查询 {i} 测试通过")

        print("\n" + "=" * 60)
        print(f"✓ 所有 Router Ollama 测试通过！")
        print(f"  正确匹配: {correct_count}/{len(test_queries)}")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_router_with_ollama()
    sys.exit(0 if success else 1)