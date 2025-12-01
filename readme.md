# RAG查询路由和分解框架

这是一个灵活的RAG（检索增强生成）系统，具有查询分解和查询路由功能，支持多种RAG策略（无RAG、朴素RAG、图RAG）。

## 概述

该框架旨在创建一个灵活的RAG（检索增强生成）系统，具有两个关键组件：
1. 查询分解 - 将复杂查询分解为子查询
2. 查询路由 - 决定如何处理每个子查询（无RAG、朴素RAG或图RAG）

该框架设计为可在每个组件的不同实现之间轻松切换，以实现性能和成本的权衡。

## 架构组件

### 1. 查询分解器模块 (decomposer/)
- 负责将复杂查询拆分为子查询
- 当前实现：基于LLM API调用
- 接口：遵循 `DecomposerInterface`

### 2. 查询路由器模块 (router/)
- 确定如何处理每个子查询
- 选项：无RAG（直接响应）、朴素RAG、图RAG
- 当前实现：基于LLM API调用
- 接口：遵循 `RouterInterface`

### 3. RAG实现 (rag_implementations/)
- 包含不同的RAG策略：
  - 朴素RAG (naive_rag/) - 基于LlamaIndex实现
  - 图RAG (graph_rag/) - 基于微软GraphRAG实现
- 模块化设计以允许交换实现
- 接口：遵循 `RAGInterface`

### 4. 核心编排器 (core/)
- 主管道，结合所有组件
- 处理流程：输入查询 → 分解 → 路由 → 执行 → 合并结果
- 包含：`orchestrator.py`

### 5. 配置系统 (config/)
- 定义每个组件的设置
- 允许在实现之间切换
- 包含：`config.py`, `settings.yaml`

### 6. 抽象接口 (interfaces/)
- 定义每种组件类型的通用接口
- 确保可交换的实现
- 包含：`decomposer_interface.py`, `router_interface.py`, `rag_interface.py`

### 7. 工具 (utils/)
- 系统的通用工具
- 包含：实用函数、辅助类

## 安装

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置API密钥和其他设置：
   - 编辑 `config/settings.yaml` 文件以配置API端点、密钥和其他参数

## 使用

1. 配置系统参数：在 `config/settings.yaml` 中设置各组件的API参数
2. 运行主程序：通过主入口文件启动系统
3. 系统将根据配置自动选择适当的RAG策略处理传入查询

## 配置

所有配置都在 `config/settings.yaml` 中进行，支持以下配置项：
- 分解器API设置
- 路由器API设置
- Naive RAG设置（API、模型、嵌入模型等）
- Graph RAG设置（API、模型、嵌入模型等）
- 系统级设置

## 扩展性

该框架设计为高度可扩展：
- 新的分解器实现可以轻松添加并遵循 `DecomposerInterface`
- 新的路由器实现可以轻松添加并遵循 `RouterInterface`
- 新的RAG实现可以轻松添加并遵循 `RAGInterface`
- 通过配置文件可灵活切换不同实现