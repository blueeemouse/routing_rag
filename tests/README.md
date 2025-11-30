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
  - 真实的router API调用
- **注意**: 需要在settings.yaml中配置有效的API密钥

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