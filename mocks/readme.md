# Mocks 组件

此目录包含项目中用于测试的模拟组件实现。

## 组件列表

- `mock_components.py`: 包含所有模拟组件的实现
  - `MockDecomposer`: 模拟查询分解器
  - `MockRouter`: 模拟查询路由器
  - `MockRAG`: 模拟RAG实现基类
  - `MockNaiveRAG`: 模拟Naive RAG实现
  - `MockGraphRAG`: 模拟Graph RAG实现
  - `MockNoRAG`: 模拟No RAG实现（直接返回答案）

## 用途

这些mock组件用于:
- 验证orchestrator的架构和流程
- 在不依赖真实API的情况下测试组件集成
- 提供可预测的测试结果
- 降低测试复杂性

## 接口兼容性

所有mock组件都实现了对应的接口：
- `MockDecomposer` 实现 `DecomposerInterface`
- `MockRouter` 实现 `RouterInterface`
- `MockRAG` 及其子类实现 `RAGInterface`

这确保了mock组件可以无缝替换为真实组件。