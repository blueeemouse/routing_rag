GraphRAG vLLM 配置模板
========================

此目录包含使用 vLLM 作为 LLM 后端的 GraphRAG 配置模板。

使用步骤：
1. 安装 vLLM: pip install vllm
2. 启动 vLLM 服务（Chat 模型）:
   vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
3. 启动 vLLM 服务（Embedding 模型，可选单独部署）:
   vllm serve BAAI/bge-small-en-v1.5 --port 8001
4. 将输入数据放入 input/ 目录
5. 修改 graphrag_vllm_config.yml 中的路径和模型名称
6. 运行 GraphRAG 索引构建

注意事项：
- vLLM 默认端口为 8000
- 模型名称使用 HuggingFace 格式（如 Qwen/Qwen2.5-3B-Instruct）
- API Key 可以填任意值（如 "EMPTY"）
- 如需调整模型，请修改配置文件中的 models 部分
