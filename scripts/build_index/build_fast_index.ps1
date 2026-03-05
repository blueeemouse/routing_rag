# GraphRAG Fast 模式索引构建脚本
# 使用 Fast 模式构建 GraphRAG 索引

# 脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 项目根目录
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

# 配置参数
$WorkDir = "D:\Develop\all_RAG\routing_rag\graphrag_index_hotpotqa_train_5000_samples_fast"
$ConfigFile = "graphrag_hotpotqa_config.yml"
$HotpotqaFile = "D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_train_v1.1_5000_samples.jsonl"
$NumSamples = 5000

# 切换到项目根目录
Set-Location $ProjectRoot

# 调用 Python 脚本
# 脚本在 scripts/build_index/ 目录下
python .\scripts\build_index\build_graphrag_index.py `
    --work_dir $WorkDir `
    --config_file $ConfigFile `
    --method fast `
    --hotpotqa_file $HotpotqaFile `
    --num_samples $NumSamples
