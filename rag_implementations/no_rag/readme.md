# No RAG 实现

此目录包含No RAG实现，它直接使用LLM进行响应，而无需检索增强。

## 组件说明

- `no_rag_impl.py`: No RAG的主要实现文件
  - 实现了RAGInterface接口
  - 直接调用LLM模型对查询进行回答
  - 适用于无需检索增强的简单查询场景

## 用途

NoRAG实现用于：
- 直接LLM响应策略
- 无需检索增强的查询
- 通用问答场景

## 接口兼容性

`NoRAG` 实现了 `RAGInterface`，确保与orchestrator和其他组件的兼容性。