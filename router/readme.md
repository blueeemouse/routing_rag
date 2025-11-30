# 查询路由器模块 (Query Router Module)

此模块确定如何处理每个子查询（无RAG、朴素RAG或图RAG）。

## 功能
- 根据子查询类型决定处理策略
- 支持多种RAG处理方式：无RAG、朴素RAG、图RAG
- 提供统一的接口供其他模块调用

## 文件结构
- `router.py` - 查询路由器的实现
- `__init__.py` - 模块初始化文件