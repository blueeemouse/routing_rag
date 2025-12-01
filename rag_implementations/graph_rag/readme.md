# 图RAG实现 (Graph RAG Implementation)

此模块实现图RAG策略，基于微软GraphRAG框架。

## 功能
- 基于知识图谱的RAG实现
- 图结构检索和生成流程
- 提供统一的RAG接口
- 支持通过API配置LLM和嵌入模型
- 需要完整的图数据结构（实体、关系、社区等）才能正常工作

## 文件结构
- `graph_rag_impl.py` - 使用微软GraphRAG的实现
- `__init__.py` - 模块初始化文件
- `readme.md` - 说明文档