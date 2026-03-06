# Patches 补丁目录

本目录存放用于修复外部依赖库兼容性问题的补丁脚本。

---

## litellm_ollama_patch.py

### 问题描述

新版 LiteLLM 与旧版 Ollama (≤0.1.45) 存在 API 兼容性问题：

| 问题 | LiteLLM 期望 | 旧版 Ollama 实际 |
|------|-------------|-----------------|
| API 端点 | `/api/embed` | `/api/embeddings` |
| 响应格式 | `{"embeddings": [[...]]}` | `{"embedding": [...]}` |

### 解决方案

修改 `litellm/llms/ollama/completion/handler.py` 文件：
1. 兼容 `embedding` 和 `embeddings` 两种响应格式
2. 使用 `/api/embeddings` 端点替代 `/api/embed`
3. 增加 `logging_obj` 方法存在性检查

### 使用方法

```bash
# 应用补丁
python patches/litellm_ollama_patch.py

# 恢复备份
python patches/litellm_ollama_patch.py restore

# 仅备份（不修改）
python patches/litellm_ollama_patch.py backup
```

### 注意事项

1. **路径配置**：补丁脚本中的路径是硬编码的，在新环境使用前需修改：
   ```python
   HANDLER_FILE = "/path/to/your/env/lib/python3.x/site-packages/litellm/llms/ollama/completion/handler.py"
   BACKUP_DIR = "/path/to/your/project/patches/backups"
   ```

2. **备份位置**：备份文件默认保存在 `/home/lhz/code/litellm_ollama_patch/backups/`


---

## 其他说明

- 升级 LiteLLM 或 Ollama 版本后，可能需要重新应用补丁或检查是否仍需补丁
- 建议在虚拟环境中使用，避免影响其他项目
