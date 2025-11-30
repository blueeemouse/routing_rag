# RAG Query Routing and Decomposition Framework Plan

## Overview
This project aims to create a flexible RAG (Retrieval-Augmented Generation) system with two key components:
1. Query Decomposition - breaking complex queries into sub-queries
2. Query Routing - deciding how to process each sub-query (no RAG, naive RAG, or graph RAG)

The framework will be designed for easy switching between different implementations of each component.

## Architecture Components

### 1. Query Decomposer Module (`decomposer/`)
- Responsible for splitting complex queries into sub-queries
- Initial implementation: LLM API call-based
- Interface should allow for future specialized models
- Contains: `decomposer.py`, `__init__.py`, `readme.md`

### 2. Query Router Module (`router/`)
- Determines how to process each sub-query
- Options: no RAG (direct response), naive RAG, graph RAG
- Initial implementation: LLM API call-based
- Interface should allow for future trained routers
- Contains: `router.py`, `__init__.py`, `readme.md`

### 3. RAG Implementations (`rag_implementations/`)
- Contains different RAG strategies:
  - Naive RAG (`naive_rag/`)
  - Graph RAG (`graph_rag/`)
- Modular design to allow swapping implementations
- Each contains: implementation files, `__init__.py`, `readme.md`

### 4. Core Orchestrator (`core/`)
- Main pipeline that combines all components
- Handles the flow: Input Query → Decompose → Route → Execute → Combine Results
- Contains: `orchestrator.py`, `__init__.py`, `readme.md`

### 5. Configuration System (`config/`)
- Defines settings for each component
- Allows switching between implementations
- Contains: `config.py`, `settings.yaml`, `readme.md`

### 6. Abstraction Interfaces (`interfaces/`)
- Defines common interfaces for each component type
- Ensures swappable implementations
- Contains: `decomposer_interface.py`, `router_interface.py`, `rag_interface.py`, `readme.md`

### 7. Utils (`utils/`)
- Common utilities for the system
- Contains: utility functions, helper classes, `readme.md`

## Implementation Phases

### Phase 1: Basic Framework
- Define interfaces for each component
- Create basic implementations with LLM API calls
- Build minimal orchestrator
- Set up configuration system

### Phase 2: Integration
- Connect all components through defined interfaces
- Test basic functionality
- Ensure modularity and swappability

### Phase 3: Documentation and Testing
- Complete all README files
- Add comprehensive test suite
- Document extension points for future implementations

---

# RAG查询路由和分解框架规划

## 概述
该项目旨在创建一个灵活的RAG（检索增强生成）系统，具有两个关键组件：
1. 查询分解 - 将复杂查询分解为子查询
2. 查询路由 - 决定如何处理每个子查询（不用RAG、朴素RAG或图RAG）

该框架将设计为可在每个组件的不同实现之间轻松切换。

## 架构组件

### 1. 查询分解器模块 (`decomposer/`)
- 负责将复杂查询拆分为子查询
- 初始实现：基于LLM API调用
- 接口应允许未来专门的模型
- 包含：`decomposer.py`, `__init__.py`, `readme.md`

### 2. 查询路由器模块 (`router/`)
- 确定如何处理每个子查询
- 选项：无RAG（直接响应）、朴素RAG、图RAG
- 初始实现：基于LLM API调用
- 接口应允许未来训练的路由器
- 包含：`router.py`, `__init__.py`, `readme.md`

### 3. RAG实现 (`rag_implementations/`)
- 包含不同的RAG策略：
  - 朴素RAG (`naive_rag/`)
  - 图RAG (`graph_rag/`)
- 模块化设计以允许交换实现
- 每个包含：实现文件、`__init__.py`、`readme.md`

### 4. 核心编排器 (`core/`)
- 主管道，结合所有组件
- 处理流程：输入查询 → 分解 → 路由 → 执行 → 合并结果
- 包含：`orchestrator.py`, `__init__.py`, `readme.md`

### 5. 配置系统 (`config/`)
- 为每个组件定义设置
- 允许在实现之间切换
- 包含：`config.py`, `settings.yaml`, `readme.md`

### 6. 抽象接口 (`interfaces/`)
- 为每种组件类型定义通用接口
- 确保可交换的实现
- 包含：`decomposer_interface.py`, `router_interface.py`, `rag_interface.py`, `readme.md`

### 7. 工具 (`utils/`)
- 系统的通用工具
- 包含：实用函数、辅助类、`readme.md`

## 实现阶段

### 第一阶段：基本框架
- 为每个组件定义接口
- 使用LLM API调用创建基本实现
- 构建最小编排器
- 设置配置系统

### 第二阶段：集成
- 通过定义的接口连接所有组件
- 测试基本功能
- 确保模块化和可交换性

### 第三阶段：文档和测试
- 完成所有README文件
- 添加全面的测试套件
- 记录未来实现的扩展点