# ============================================
# End-to-End Router Test Script (Mode 1: NoRAG + NaiveRAG)
# ============================================
#
# Features:
# - Load trained Router model
# - Route HotpotQA test data
# - Execute NoRAG and NaiveRAG strategies
# - Evaluate metrics (EM, F1, retrieval_time, generation_time)
#
# Usage:
#   .\run_router_e2e_test.ps1
#   Or specify sample count:
#   .\run_router_e2e_test.ps1 -NumSamples 100
#
# ============================================

param(
    [int]$NumSamples = 1000,
    [string]$OutputFile = "results/router_eval_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
)

# Switch to project root directory
$ProjectRoot = "D:\Develop\all_RAG\routing_rag"
Set-Location $ProjectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "End-to-End Router Test (NoRAG + NaiveRAG)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Project Root: $ProjectRoot" -ForegroundColor White
Write-Host "  Sample Count: $NumSamples" -ForegroundColor White
Write-Host "  Output File: $OutputFile" -ForegroundColor White
Write-Host ""

# Check required files
Write-Host "Checking required files..." -ForegroundColor Yellow

$ModelPath = "router_models\tie_weight_search\all-MiniLM-L6-v2\tie_weight_1\checkpoint_best_val"
$HotpotqaFile = "HotpotQA\hotpot_dev_distractor_1000_samples.jsonl"
$NaiveRagIndex = "naive_rag_index_hotpotqa_1000_samples"

if (-not (Test-Path $ModelPath)) {
    Write-Host "ERROR: Router model path not found: $ModelPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $HotpotqaFile)) {
    Write-Host "ERROR: HotpotQA data file not found: $HotpotqaFile" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $NaiveRagIndex)) {
    Write-Host "ERROR: NaiveRAG index not found: $NaiveRagIndex" -ForegroundColor Red
    exit 1
}

Write-Host "  [OK] Router Model: $ModelPath" -ForegroundColor Green
Write-Host "  [OK] HotpotQA Data: $HotpotqaFile" -ForegroundColor Green
Write-Host "  [OK] NaiveRAG Index: $NaiveRagIndex" -ForegroundColor Green
Write-Host ""

# Create output directory
$OutputDir = Split-Path $OutputFile -Parent
if ($OutputDir -and -not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

# Run test
Write-Host "Starting test..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

$StartTime = Get-Date

python tests/test_router_e2e.py `
    --model_path $ModelPath `
    --hotpotqa_file $HotpotqaFile `
    --naive_rag_index_path $NaiveRagIndex `
    --max_samples $NumSamples `
    --output $OutputFile

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Duration: $($Duration.TotalMinutes.ToString('F2')) minutes" -ForegroundColor White
Write-Host "Results File: $OutputFile" -ForegroundColor White
Write-Host ""

# Display summary if output file exists
if (Test-Path $OutputFile) {
    Write-Host "Performance Summary:" -ForegroundColor Yellow
    
    # Use Python to read JSON and display summary
    $PythonScript = @"
import json
import sys

try:
    with open('$OutputFile', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    overall = metrics.get('overall', {})
    
    print(f"  Samples: {overall.get('num_samples', 0)}")
    print(f"  EM: {overall.get('em', 0):.4f}")
    print(f"  F1: {overall.get('f1', 0):.4f}")
    print(f"  Avg Total Time: {overall.get('avg_total_time', 0):.4f}s")
    print(f"  Avg Retrieval Time: {overall.get('avg_retrieval_time', 0):.4f}s")
    print(f"  Avg Generation Time: {overall.get('avg_generation_time', 0):.4f}s")
    print()
    
    by_strategy = metrics.get('by_strategy', {})
    print('Strategy Statistics:')
    for strategy, stats in by_strategy.items():
        print(f"  {strategy}:")
        print(f"    Count: {stats.get('count', 0)} ({stats.get('ratio', 0):.2%})")
        print(f"    EM: {stats.get('em', 0):.4f}, F1: {stats.get('f1', 0):.4f}")
        print(f"    Avg Time: {stats.get('avg_total_time', 0):.4f}s")
except Exception as e:
    print(f"Error reading results: {e}", file=sys.stderr)
    sys.exit(1)
"@
    
    $PythonScript | python
}

Write-Host ""
Write-Host "Tip: You can customize sample count with -NumSamples parameter" -ForegroundColor Gray
Write-Host "      Default is 1000 samples (already configured in script)" -ForegroundColor Gray
Write-Host "      Example: .\run_router_e2e_test.ps1 -NumSamples 100" -ForegroundColor Gray
