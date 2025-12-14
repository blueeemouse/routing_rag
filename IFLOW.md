# RAG查询路由和分解框架

## 项目概述

这是一个灵活的RAG（检索增强生成）系统，具有查询分解和查询路由功能，支持多种RAG策略（无RAG、朴素RAG、图RAG）。该框架旨在创建一个高性能、模块化的系统，可以在不同组件实现之间轻松切换，以实现性能和成本的权衡。

## 核心架构

### 1. 查询分解器 (decomposer/)
- 负责将复杂查询拆分为子查询
- 当前实现：基于LLM API调用
- 接口：遵循 `DecomposerInterface`
- 配置：支持自定义prompt和模型选择

### 2. 查询路由器 (router/)
- 确定如何处理每个子查询
- 选项：无RAG（直接响应）、朴素RAG、图RAG
- 当前实现：基于LLM API调用
- 接口：遵循 `RouterInterface`

### 3. RAG实现 (rag_implementations/)
- **朴素RAG** (`naive_rag/`) - 基于LlamaIndex实现
- **图RAG** (`graph_rag/`) - 基于微软GraphRAG实现
- **无RAG** (`no_rag/`) - 直接LLM响应实现
- 模块化设计以允许交换实现
- 接口：遵循 `RAGInterface`

### 4. 核心编排器 (core/)
- 主管道，结合所有组件
- 处理流程：输入查询 → 分解 → 路由 → 执行 → 合并结果
- 支持上下文传递
- 包含：`orchestrator.py`

### 5. 配置系统 (config/)
- 定义每个组件的设置
- 允许在实现之间切换
- 支持环境变量配置
- 包含：`config.py`, `settings.yaml`

### 6. 抽象接口 (interfaces/)
- 定义每种组件类型的通用接口
- 确保可交换的实现
- 包含：`decomposer_interface.py`, `router_interface.py`, `rag_interface.py`

## 技术栈

- **Python** - 主要编程语言
- **LlamaIndex** - 朴素RAG实现
- **微软GraphRAG** - 图RAG实现
- **LiteLLM** - 统一LLM API调用
- **PyYAML** - 配置文件解析
- **OpenAI API** - 默认LLM和嵌入模型

## 安装和设置

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥
1. 复制 `.env.example` 为 `.env`
2. 编辑 `.env` 文件，添加你的API密钥：
   ```
   API_BASE_URL=https://api.openai.com/v1
   DECOMPOSER_API_KEY=your_api_key
   ROUTER_API_KEY=your_api_key
   NAIVE_RAG_API_KEY=your_api_key
   GRAPHRAG_API_KEY=your_api_key
   ```

### 3. 加载环境变量（Windows）
```powershell
# 在项目根目录执行
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
```

### 4. 配置系统参数
编辑 `config/settings.yaml` 文件以配置：
- 各组件的API参数
- 模型选择
- 检索参数（chunk_size、top_k等）
- 策略启用/禁用

## 使用方法

### 基本使用
```python
from core.orchestrator import Orchestrator
from config.config import load_config
from rag_implementations.naive_rag.naive_rag_impl import NaiveRAG
from rag_implementations.graph_rag.graph_rag_impl import GraphRAG
from rag_implementations.no_rag.no_rag_impl import NoRAG
from decomposer.decomposer import LLMDecomposer
from router.router import LLMRouter

# 加载配置
config = load_config()

# 初始化组件
decomposer = LLMDecomposer(config['decomposer'])
router = LLMRouter(config['router'])
rag_implementations = {
    'no_rag': NoRAG(config['no_rag'] if 'no_rag' in config else {}),
    'naive_rag': NaiveRAG(config['naive_rag']),
    'graph_rag': GraphRAG(config['graph_rag'])
}

# 创建编排器
orchestrator = Orchestrator(decomposer, router, rag_implementations)

# 处理查询
result = orchestrator.process_query("你的复杂查询")
print(result)
```

### 索引构建
```python
# NaiveRAG索引构建
naive_rag = NaiveRAG(config['naive_rag'])
naive_rag.build_index_from_data(documents)

# GraphRAG索引构建
graph_rag = GraphRAG(config['graph_rag'])
graph_rag.build_index_from_path("/path/to/data", "/path/to/output")
```

## 测试

项目包含全面的测试套件：
- 单元测试：各组件独立功能测试
- 集成测试：完整流程测试
- 模拟测试：使用模拟组件验证架构

运行测试：
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python tests/test_orchestrator_real_components.py
```

## 项目状态

### 已完成功能
- [x] 基础架构搭建
- [x] 查询分解器实现
- [x] 查询路由器实现
- [x] NaiveRAG实现
- [x] GraphRAG实现
- [x] NoRAG实现
- [x] 核心编排器实现
- [x] 配置管理系统
- [x] 索引构建与查询解耦
- [x] 端到端集成测试

### 正在进行
- [ ] GraphRAG其他搜索模式实现
- [ ] 性能优化

### 计划中
- [ ] 更多RAG策略实现
- [ ] 高级路由策略
- [ ] 结果合并策略优化
- [ ] 监控和日志系统

## 扩展性

该框架设计为高度可扩展：

### 添加新的分解器实现
1. 实现 `DecomposerInterface` 接口
2. 在 `decomposer/` 目录下创建新模块
3. 在配置文件中添加相应配置

### 添加新的路由器实现
1. 实现 `RouterInterface` 接口
2. 在 `router/` 目录下创建新模块
3. 在配置文件中添加相应配置

### 添加新的RAG实现
1. 实现 `RAGInterface` 接口
2. 在 `rag_implementations/` 目录下创建新子目录
3. 在配置文件中添加相应配置
4. 在编排器中注册新实现

## 配置说明

所有配置都在 `config/settings.yaml` 中进行，支持以下配置项：

- **decomposer**: 分解器API设置
- **router**: 路由器API设置
- **naive_rag**: Naive RAG设置（API、模型、嵌入模型等）
- **graph_rag**: Graph RAG设置（API、模型、嵌入模型等）
- **rag**: 全局RAG设置（启用/禁用各种策略）

配置支持环境变量替换，格式为 `${VARIABLE_NAME}`。

## 常见问题

### Q: 如何切换不同的LLM提供商？
A: 在 `config/settings.yaml` 中修改 `api_url` 和相应的 `api_key`，支持OpenAI、DeepSeek等兼容OpenAI API格式的提供商。

### Q: 如何添加自定义文档数据？
A: 对于NaiveRAG，使用 `build_index_from_data` 方法传入文档列表；对于GraphRAG，使用 `build_index_from_path` 方法指定数据目录路径。

### Q: 如何优化查询性能？
A: 可以通过调整 `chunk_size`、`top_k` 等参数，或者选择更快的模型来优化性能。

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 发送邮件至项目维护者

---

*最后更新: 2025年12月13日*