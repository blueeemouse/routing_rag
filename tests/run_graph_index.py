import os
import sys
from pathlib import Path

# 打印当前路径以便调试
# print("Current working directory:", os.getcwd())
# print("Python path:", sys.path[:3])  # 打印前三个路径

# 添加graphrag到路径 - 需要添加到正确位置
GRAPH_RAG_PATH = 'D:/Develop/all_RAG/routing_rag/graphrag'
if GRAPH_RAG_PATH not in sys.path:
    sys.path.insert(0, GRAPH_RAG_PATH)

# 设置环境变量
os.environ['GRAPHRAG_API_KEY'] = os.getenv('GRAPHRAG_API_KEY', 'YOUR_API_KEY_HERE')

# 验证导入（验证通过）
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
# print(f"index_cli function exists: {index_cli is not None}")
# print(f"Available indexing methods: {[method.value for method in IndexingMethod]}")

# ###################################
# # 临时添加调试代码（放在 index_cli 调用前），验证路径和文件
# input_dir = Path("D:/Develop/all_RAG/routing_rag/HotpotQA/raw")
# print(f"输入目录是否存在：{input_dir.exists()}")
# print(f"目录下的文件：{list(input_dir.glob('*'))}")  # 打印所有文件
# print(f"JSON文件数量：{len(list(input_dir.glob('*.json'))) + len(list(input_dir.glob('*.jsonl')))}")
# ###################################

####################################
# # 调试：手动加载转换后的数据
# from graphrag.index.workflows.load_input_documents import load_input_documents
# from graphrag.config import InputConfig

# input_config = InputConfig(
#     type="file",
#     file_type="json",
#     base_dir="D:/Develop/all_RAG/routing_rag/HotpotQA/raw",
#     # file_pattern="hotpotqa_processed.jsonl",
#     encoding="utf-8",
#     text_column="context",
#     json_format="jsonl"
# )

# try:
#     load_result = load_input_documents(config=input_config)
#     print(f"✅ 成功加载文档数：{len(load_result.result)}")
#     print(f"✅ 第一条文档的文本预览：\n{load_result.result['context'].iloc[0][:200]}...")
# except Exception as e:
#     print(f"❌ 加载失败：{e}")
#     import traceback
#     traceback.print_exc()
####################################

# 执行索引
try:
    index_cli(
        root_dir=Path("D:/Develop/all_RAG/routing_rag/HotpotQA"),
        verbose=True,
        memprofile=False,
        cache=True,
        config_filepath=Path("D:/Develop/all_RAG/routing_rag/HotpotQA/graphrag_config_final.yml"),
        dry_run=False,  # 设置为False以实际运行索引
        skip_validation=False,
        output_dir=None,  # 使用配置文件中的默认值
        method=IndexingMethod.Standard  # 使用标准索引方法
    )
    print("Indexing process completed successfully.")
except Exception as e:
    print(f"Error during indexing: {e}")
    import traceback
    traceback.print_exc()