"""
阶段0：验证 Ollama 环境是否就绪
"""
import requests
import json

def test_ollama_connection():
    """测试 Ollama 服务连接"""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama 服务连接成功")
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            print(f"✓ 已安装模型: {models}")
            return True, models
        else:
            print(f"✗ Ollama 服务响应错误: {response.status_code}")
            return False, []
    except Exception as e:
        print(f"✗ 连接 Ollama 失败: {e}")
        return False, []

def test_ollama_generate(model_name):
    """测试 Ollama 模型生成"""
    try:
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": "你好",
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 模型生成测试成功 ({model_name})")
            print(f"  响应: {result.get('response', '')[:50]}...")
            return True
        else:
            print(f"✗ 模型生成失败 ({model_name}): {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 模型生成测试失败 ({model_name}): {e}")
        return False

def test_ollama_chat_completions(model_name):
    """测试 OpenAI 兼容的 Chat Completions API"""
    try:
        url = "http://127.0.0.1:11434/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            message = result.get('choices', [{}])[0].get('message', {})
            print(f"✓ OpenAI 兼容 API 测试成功 ({model_name})")
            print(f"  响应: {message.get('content', '')[:50]}...")
            return True
        else:
            print(f"✗ OpenAI 兼容 API 失败 ({model_name}): {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ OpenAI 兼容 API 测试失败 ({model_name}): {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("阶段0：Ollama 环境验证")
    print("=" * 60)

    # 测试连接
    print("\n1. 连接测试:")
    success, models = test_ollama_connection()

    if not success:
        print("\n✗ 无法连接到 Ollama 服务，请确保服务已启动")
        exit(1)

    # 检查是否有 Qwen 模型
    qwen_models = [m for m in models if 'qwen' in m.lower()]
    if qwen_models:
        test_model = qwen_models[0]
        print(f"\n✓ 找到 Qwen 模型: {test_model}")
    else:
        print(f"\n⚠ 未找到 Qwen 模型")
        print(f"  建议下载: ollama pull qwen2.5:3b")
        print(f"  或使用现有模型测试: {models[0] if models else '无'}")
        if models:
            test_model = models[0]
            print(f"  将使用 {test_model} 进行测试")
        else:
            print("✗ 没有可用的模型")
            exit(1)

    # 测试生成
    print(f"\n2. 生成测试 ({test_model}):")
    gen_success = test_ollama_generate(test_model)

    # 测试 OpenAI 兼容 API
    print(f"\n3. OpenAI 兼容 API 测试 ({test_model}):")
    api_success = test_ollama_chat_completions(test_model)

    # 总结
    print("\n" + "=" * 60)
    if gen_success and api_success:
        print("✓ 所有测试通过！阶段0完成！")
        print("✓ 可以进入阶段1：最小改动验证")
        print(f"✓ 推荐模型: {test_model}")
    else:
        print("✗ 部分测试失败，请检查 Ollama 服务")
    print("=" * 60)
