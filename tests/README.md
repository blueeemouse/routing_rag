# 测试说明文档

## 项目测试结构

本项目包含以下测试文件：

### 1. test_decomposer.py

- **目的**: 测试decomposer模块的功能
- **测试内容**:
  - 查询分解功能
  - API调用模拟
  - 边界情况处理
- **依赖**: decomposer模块、DecomposerInterface

### 2. test_router.py

- **目的**: 测试router模块的功能
- **测试内容**:
  - 查询路由功能
  - API调用模拟
  - 策略识别
- **依赖**: router模块、RouterInterface

### 3. test_api_calls.py (功能测试脚本)

- **目的**: 验证实际API调用是否正常
- **测试内容**:

  - 真实的decomposer API调用
  - 真实的router API调用(前两部分的测试, 因为需要打开 `config/settings.yaml`, 故需要在routing_rag路径下运行 `tests/test_api_calls.py`
  - API调用版本的naive_rag
    - 通过在这里的测试结果, 我们可以发现, 给定的模拟数据不同的时候, 答复也是不同的. 由此进一步证明了确实有在检索增强生成
  - API调用版本的微软graphrag

### 4. test_graph_rag.py

- **目的**: 测试graphrag类的各个函数(目的是好的, 但现在可以说是一点没测试到...)

### 5. quick_test_graph_index.py

- **目的**: 快速测试能不能通过调用项目根目录下的微软graphrag的函数, 实现建立索引

### 6. quick_litellm_api_test.py

- **目的**: 测试litellm调用api(主要是确认下格式. 这个之所以需要测试, 是因为之前跑LlamaIndex的时候, 遇到报错, 似乎说是LlamaIndex内部是用litellm调用api的

### 7. test_graphrag_import.py

- **目的**: 测试GraphRAG库的导入功能
- **测试内容**:
  - 验证微软GraphRAG库组件是否能正确导入
  - 检查支持的搜索模式是否可用
  - 验证库的可用性状态
- **依赖**: graphrag_impl模块

### 8. test_graphrag_config.py

- **目的**: 测试GraphRAG的配置加载功能
- **测试内容**:
  - 验证配置参数从settings.yaml正确加载
  - 检查API URL、密钥、模型等参数
  - 验证配置的一致性
- **依赖**: graphrag_impl模块、config模块

### 9. test_graphrag_build_index.py

- **目的**: 测试GraphRAG类里面的build_index方法
- **测试内容**:
  - 验证build_index方法接口的可用性
  - 测试方法参数验证
  - 验证索引构建功能的基本调用
- **依赖**: graphrag_impl模块

### 10. test_graphrag_local_search.py

- **目的**: 测试GraphRAG的_local_search方法
- **测试内容**:
  - 验证本地搜索功能的可用性
  - 测试数据加载和查询执行
  - 验证与已构建索引的集成
- **依赖**: graphrag_impl模块

### 11. test_graphrag_execute_integration.py

- **目的**: 测试GraphRAG的execute方法与_local_search的集成
- **测试内容**:
  - 验证execute方法能够正确调用_local_search
  - 测试端到端查询流程
  - 验证集成后的功能完整性
- **依赖**: graphrag_impl模块

### 12. test_naive_rag_refactor.py

- **目的**: 测试NaiveRAG重构后的功能
- **测试内容**:
  - 验证build_index方法的功能
  - 测试execute方法使用预构建索引
  - 验证向后兼容性
- **依赖**: naive_rag_impl模块
- **说明**: 在重构NaiveRAG以实现索引构建与查询执行解耦后，验证重构功能的正确性

### 重要说明：接口更新

- **RAGInterface** 已扩展，现在包含 `build_index` 方法，以支持解耦的索引构建与查询执行
- **NaiveRAG** 已更新以实现新的接口，支持独立的索引构建
- **GraphRAG** 当前的 `build_index` 方法与新接口不完全兼容，将在后续更新中使其兼容
- 现有测试文件（如 `test_naive_rag.py` 和 `test_api_calls.py`）仍可正常运行，因为保持了向后兼容性

## 配置说明

### 独立配置结构

- `decomposer` 部分包含decomposer模块的独立配置
- `router` 部分包含router模块的独立配置
- 每个模块都有自己的 `api_url`、`api_key` 和 `model`

### 配置文件路径

- 配置文件: `config/settings.yaml`
- 配置管理类: `config/config.py`

## 运行测试

### 单元测试

```bash
cd tests
python -m pytest test_decomposer.py -v
python -m pytest test_router.py -v
```

或者运行所有测试：

```bash
cd tests
python -m pytest . -v
```

### 功能测试（API调用验证）

```bash
python test_api_calls.py
```

## 注意事项

1. **Mock测试**: `test_decomposer.py` 和 `test_router.py` 使用mock来模拟API调用
2. **实际测试**: `test_api_calls.py` 进行真实的API调用，需要有效的API密钥
3. **环境**: 建议在 `ant-graphrag-dev` 环境中运行测试
4. **依赖**: 确保已安装 `requests` 和 `PyYAML`

## 测试用例说明

### test_decomposer.py 中的测试用例

- `test_decompose_query`: 测试正常查询分解
- `test_decompose_empty_response`: 测试空响应情况

### test_router.py 中的测试用例

- `test_route_to_no_rag`: 测试路由到no_rag策略
- `test_route_to_naive_rag`: 测试路由到naive_rag策略
- `test_route_to_graph_rag`: 测试路由到graph_rag策略
- `test_route_with_api_error`: 测试API错误时的降级处理
- `test_route_with_unrecognized_response`: 测试无法识别响应时的处理

## API调用验证脚本说明

`test_api_calls.py` 脚本会:

1. 初始化decomposer并测试查询分解
2. 初始化router并测试查询路由
3. 验证API调用是否成功
4. 展示详细的测试结果和错误信息
