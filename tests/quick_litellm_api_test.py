import requests
import json
import os

# 测试API端点是否可用
def test_api_connectivity():
    api_base_url = os.getenv('API_BASE_URL', 'https://api.agicto.cn/v1')  # ← 修改：使用环境变量
    api_url = f"{api_base_url}/chat/completions"  # ← 修改：动态构建URL
    api_key = os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE')

    # 简单的模型列表查询
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 测试API连接
    try:
        response = requests.get(
            f"{api_base_url}/models",  # ← 修改：使用环境变量
            headers=headers,
            timeout=10  # 设置较短的超时时间
        )
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            model_ids = [model.get('id', '') for model in models.get('data', [])]
            print(f"Available models count: {len(model_ids)}")
            # 只出包含embedding的模型
            embedding_models = [mid for mid in model_ids if 'embedding' in mid.lower()]
            print(f"Embedding models: {embedding_models}")
        else:
            print(f"Models endpoint error: {response.text}")
    except requests.exceptions.Timeout:
        print("Timeout when connecting to API models endpoint")
    except requests.exceptions.RequestException as e:
        print(f"Request error when connecting to API: {e}")

    # 测试聊天完成接口
    test_payload = {
        "model": "deepseek-v3",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=test_payload,
            timeout=10
        )
        print(f"Chat completion endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("API connection test successful!")
            result = response.json()
            print(f"Response preview: Response received successfully")
        else:
            print(f"API connection failed with status {response.status_code}")
            print(f"Error: {response.text}")
    except requests.exceptions.Timeout:
        print("Timeout when testing chat completion endpoint")
    except requests.exceptions.RequestException as e:
        print(f"Request error when testing chat completion: {e}")

def test_with_litellm():
    """测试使用litellm库的调用"""
    try:
        import litellm
        from litellm import completion

        # 测试API调用 - 使用正确的deepseek前缀
        api_base_url = os.getenv('API_BASE_URL', 'https://api.agicto.cn/v1')  # ← 修改：使用环境变量
        response = completion(
            model="deepseek/deepseek-v3",
            messages=[{"role": "user", "content": "Hello"}],
            api_base=f"{api_base_url}/chat/completions",  # ← 修改：动态构建URL
            api_key=os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE'),
            timeout=10
        )
        print("Litellm API test successful!")
        print(f"Response content: {response.choices[0].message.content}")

    except Exception as e:
        print(f"Litellm API test failed: {e}")
        import traceback
        traceback.print_exc()

def test_embedding_with_litellm():
    """测试embedding模型调用"""
    try:
        import litellm
        from litellm import embedding

        # 测试embedding API调用
        api_base_url = os.getenv('API_BASE_URL', 'https://api.agicto.cn/v1')  # ← 修改：使用环境变量
        response = embedding(
            model="text-embedding-ada-002",
            input=["Hello world", "Goodbye world"],
            api_base=api_base_url,  # ← 修改：使用环境变量
            api_key=os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE'),
            timeout=10
        )
        print("Litellm Embedding API test successful!")
        print(f"Embedding response received for {len(response.data)} items")
        print(f"First embedding vector length: {len(response.data[0]['embedding']) if response.data else 0}")

    except Exception as e:
        print(f"Litellm Embedding API test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing API connectivity...")
    test_api_connectivity()
    print("\nTesting with Litellm for chat model...")
    test_with_litellm()
    print("\nTesting with Litellm for embedding model...")
    test_embedding_with_litellm()