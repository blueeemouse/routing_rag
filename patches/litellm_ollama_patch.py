"""
litellm Ollama 兼容性补丁
修复新版 litellm 与旧版 Ollama (0.1.45) 的兼容性问题

问题：
1. litellm 使用 /api/embed 端点，旧版 Ollama 只有 /api/embeddings
2. litellm 期望响应中有 "embeddings" 键，旧版 Ollama 返回 "embedding"
3. litellm 假设 logging_obj 有 debug/warning 方法，但 graphrag 传递的对象可能没有

修改位置：
/home/lhz/miniconda3/envs/rag_routing/lib/python3.10/site-packages/litellm/llms/ollama/completion/handler.py

修改内容：
1. _process_ollama_embedding_response 函数：支持 "embedding" 和 "embeddings" 两种响应格式
2. ollama_aembeddings 函数：使用 /api/embeddings 端点
3. ollama_embeddings 函数：使用 /api/embeddings 端点
"""

import os
import shutil
from datetime import datetime

# 目标文件路径
HANDLER_FILE = "/home/lhz/miniconda3/envs/rag_routing/lib/python3.10/site-packages/litellm/llms/ollama/completion/handler.py"
BACKUP_DIR = "/home/lhz/code/litellm_ollama_patch/backups"


def backup_original():
    """备份原始文件"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"handler_{timestamp}.py.bak")
    shutil.copy2(HANDLER_FILE, backup_file)
    print(f"已备份原始文件到: {backup_file}")
    return backup_file


def restore_from_backup(backup_file=None):
    """从备份恢复"""
    if backup_file is None:
        # 找最新的备份
        backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.bak')])
        if not backups:
            print("没有找到备份文件")
            return False
        backup_file = os.path.join(BACKUP_DIR, backups[-1])
    
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, HANDLER_FILE)
        print(f"已从备份恢复: {backup_file}")
        return True
    else:
        print(f"备份文件不存在: {backup_file}")
        return False


def apply_patch():
    """应用补丁"""
    # 备份原始文件
    backup_original()
    
    # 读取文件内容
    with open(HANDLER_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 补丁1: 修改 _process_ollama_embedding_response 函数
    old_process_func = '''def _process_ollama_embedding_response(
    response_json: dict,
    prompts: List[str],
    model: str,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: Any,
) -> EmbeddingResponse:
    output_data = []
    embeddings: List[List[float]] = response_json["embeddings"]

    for idx, emb in enumerate(embeddings):
        output_data.append({"object": "embedding", "index": idx, "embedding": emb})

    input_tokens = response_json.get("prompt_eval_count", None)

    if input_tokens is None:
        if encoding is not None:
            input_tokens = len(encoding.encode("".join(prompts)))
            if logging_obj:
                logging_obj.debug(
                    "Ollama response missing prompt_eval_count; estimated with encoding."
                )
        else:
            input_tokens = 0
            if logging_obj:
                logging_obj.warning(
                    "Missing prompt_eval_count and no encoding provided; defaulted to 0."
                )'''
    
    new_process_func = '''def _process_ollama_embedding_response(
    response_json: dict,
    prompts: List[str],
    model: str,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: Any,
) -> EmbeddingResponse:
    output_data = []
    
    # 兼容旧版 Ollama：支持 "embedding" (单数) 和 "embeddings" (复数)
    if "embeddings" in response_json:
        embeddings: List[List[float]] = response_json["embeddings"]
    elif "embedding" in response_json:
        # 旧版 Ollama 返回单个 embedding，转换为列表
        embeddings = [response_json["embedding"]]
    else:
        raise ValueError(f"Ollama response missing 'embeddings' or 'embedding' key: {response_json.keys()}")

    for idx, emb in enumerate(embeddings):
        output_data.append({"object": "embedding", "index": idx, "embedding": emb})

    input_tokens = response_json.get("prompt_eval_count", None)

    if input_tokens is None:
        if encoding is not None:
            input_tokens = len(encoding.encode("".join(prompts)))
            if logging_obj and hasattr(logging_obj, 'debug'):
                logging_obj.debug(
                    "Ollama response missing prompt_eval_count; estimated with encoding."
                )
        else:
            input_tokens = 0
            if logging_obj and hasattr(logging_obj, 'warning'):
                logging_obj.warning(
                    "Missing prompt_eval_count and no encoding provided; defaulted to 0."
                )'''
    
    content = content.replace(old_process_func, new_process_func)
    
    # 补丁2: 修改 ollama_aembeddings 函数
    old_async_func = '''async def ollama_aembeddings(
    api_base: str,
    model: str,
    prompts: List[str],
    model_response: EmbeddingResponse,
    optional_params: dict,
    logging_obj: Any,
    encoding: Any,
):
    if not api_base.endswith("/api/embed"):
        api_base += "/api/embed"'''
    
    new_async_func = '''async def ollama_aembeddings(
    api_base: str,
    model: str,
    prompts: List[str],
    model_response: EmbeddingResponse,
    optional_params: dict,
    logging_obj: Any,
    encoding: Any,
):
    # 兼容旧版 Ollama：使用 /api/embeddings 端点
    if not api_base.endswith("/api/embeddings") and not api_base.endswith("/api/embed"):
        api_base += "/api/embeddings"'''
    
    content = content.replace(old_async_func, new_async_func)
    
    # 补丁3: 修改 ollama_embeddings 函数
    old_sync_func = '''def ollama_embeddings(
    api_base: str,
    model: str,
    prompts: List[str],
    optional_params: dict,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: Any = None,
):
    if not api_base.endswith("/api/embed"):
        api_base += "/api/embed"'''
    
    new_sync_func = '''def ollama_embeddings(
    api_base: str,
    model: str,
    prompts: List[str],
    optional_params: dict,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: Any = None,
):
    # 兼容旧版 Ollama：使用 /api/embeddings 端点
    if not api_base.endswith("/api/embeddings") and not api_base.endswith("/api/embed"):
        api_base += "/api/embeddings"'''
    
    content = content.replace(old_sync_func, new_sync_func)
    
    # 写入修改后的文件
    with open(HANDLER_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("补丁已应用!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'restore':
            # 恢复备份
            restore_from_backup(sys.argv[2] if len(sys.argv) > 2 else None)
        elif sys.argv[1] == 'backup':
            # 仅备份
            backup_original()
        else:
            print("用法:")
            print("  python handler_patch.py         # 应用补丁")
            print("  python handler_patch.py restore # 恢复最新备份")
            print("  python handler_patch.py restore <backup_file> # 恢复指定备份")
            print("  python handler_patch.py backup  # 仅备份")
    else:
        # 应用补丁
        apply_patch()
