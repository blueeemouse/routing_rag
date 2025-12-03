import os
import sys
from pathlib import Path

# 添加graphrag到路径 - 需要添加到正确位置
GRAPH_RAG_PATH = 'D:/Develop/all_RAG/routing_rag/graphrag'
if GRAPH_RAG_PATH not in sys.path:
    sys.path.insert(0, GRAPH_RAG_PATH)

# # 设置环境变量
# os.environ['GRAPHRAG_API_KEY'] = os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE')

# 验证导入
try:
    from graphrag.cli.index import index_cli
    print("SUCCESS: Successfully imported index_cli from graphrag.cli.index")
except ImportError as e:
    print(f"ERROR: Failed to import index_cli: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from graphrag.config.enums import IndexingMethod
    print("SUCCESS: Successfully imported IndexingMethod from graphrag.config.enums")
except ImportError as e:
    print(f"ERROR: Failed to import IndexingMethod: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查导入的函数和类
print(f"index_cli function exists: {index_cli is not None}")
print(f"Available indexing methods: {[method.value for method in IndexingMethod]}")

# 执行索引 - 使用简化的测试数据
try:
    print("Starting quick test with simplified data...")
    index_cli(
        root_dir=Path("D:/Develop/all_RAG/routing_rag/TestData"),
        verbose=True,
        memprofile=False,
        cache=True,
        config_filepath=Path("D:/Develop/all_RAG/routing_rag/TestData/graphrag_test_config.yml"),
        dry_run=False,  # 设置为False以实际运行索引
        skip_validation=False,
        output_dir=None,  # 使用配置文件中的默认值
        method=IndexingMethod.Standard  # 使用标准索引方法
    )
    print("Quick test indexing process completed successfully.")
except Exception as e:
    print(f"Error during quick test indexing: {e}")
    import traceback
    traceback.print_exc()