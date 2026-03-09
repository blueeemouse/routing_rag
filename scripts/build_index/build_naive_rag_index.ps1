# NaiveRAG Index Build Script
# Build NaiveRAG index from HotpotQA training data

# Script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Project root directory
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

# Configuration
$HOTPOTQA_FILE = "D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_train_v1.1_10000_samples.jsonl"
$INDEX_PATH = "D:\Develop\all_RAG\naive_rag_index_storage_10000_train_samples"
$NUM_SAMPLES = 10000

# Change to project root
Set-Location $PROJECT_ROOT

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "NaiveRAG Index Build" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Project root: $PROJECT_ROOT"
Write-Host "Data file: $HOTPOTQA_FILE"
Write-Host "Index path: $INDEX_PATH"
Write-Host "Num samples: $NUM_SAMPLES"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Call Python script
python ./scripts/build_index/build_naive_rag_index.py `
    --hotpotqa_file "$HOTPOTQA_FILE" `
    --index_path "$INDEX_PATH" `
    --num_samples $NUM_SAMPLES

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Index build completed!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Index build failed!" -ForegroundColor Red
}
