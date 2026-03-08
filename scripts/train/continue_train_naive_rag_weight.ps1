# Continue Training - Boost Naive-RAG Class Weight
# 
# Purpose: Continue training to boost naive_rag class weight
#          to improve naive_rag accuracy (currently 46.51% vs no_rag 65.26%)
#
# Baseline:
#   - Val Accuracy: 61.3%
#   - no_rag accuracy: 65.26%
#   - naive_rag accuracy: 46.51%
#
# Training:
#   - Resume from: filtered_data_training/all-MiniLM-L6-v2/checkpoint_best_val
#   - Learning rate: 1e-5
#   - Epochs: 3
#   - Class weights: no_rag=1.0, naive_rag=<weight>

param(
    [string]$NaiveRagWeight = "1.2"
)

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Validate weight parameter
$validWeights = @("1.2", "1.5", "2.0")
if ($NaiveRagWeight -notin $validWeights) {
    Write-Host "ERROR: Invalid weight. Must be one of: $($validWeights -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Continue Training - Boost Naive-RAG Weight to $NaiveRagWeight" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Model config
$Backbone = "sentence-transformers/all-MiniLM-L6-v2"

# Training hyperparameters
$LearningRate = "1e-5"
$Epochs = 3
$BatchSize = 32

# Class weights: no_rag=1.0, naive_rag=$NaiveRagWeight
$ClassWeights = "no_rag=1.0,naive_rag=$NaiveRagWeight"

# Resume checkpoint
$ResumeCheckpoint = "router_models/filtered_data_training/all-MiniLM-L6-v2/checkpoint_best_val"

# Data
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_rag_strictly_better_removed.json"
$ValData = "evaluation_results/router_test_labels.json"

# Output directory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = "router_models/filtered_data_training/continue_naive_rag_weight_${NaiveRagWeight}"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Cyan
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs" -ForegroundColor Cyan
Write-Host "  Class Weights: $ClassWeights" -ForegroundColor Cyan
Write-Host "  Resume: $ResumeCheckpoint" -ForegroundColor Cyan
Write-Host ""

Write-Host "Data:" -ForegroundColor Yellow
Write-Host "  Train: $TrainData" -ForegroundColor Cyan
Write-Host "  Val: $ValData" -ForegroundColor Cyan
Write-Host ""

Write-Host "Output:" -ForegroundColor Yellow
Write-Host "  Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Check if checkpoint exists
if (-not (Test-Path $ResumeCheckpoint)) {
    Write-Host "ERROR: Checkpoint not found: $ResumeCheckpoint" -ForegroundColor Red
    exit 1
}

# Build command
$params = @(
    "router/train_router.py",
    "--config", "config/train_classification_5000.yaml",
    "--train_data", $TrainData,
    "--val_data", $ValData,
    "--backbone", $Backbone,
    "--learning_rate", $LearningRate,
    "--epochs", $Epochs.ToString(),
    "--batch_size", $BatchSize.ToString(),
    "--class_weights", $ClassWeights,
    "--resume", $ResumeCheckpoint,
    "--output_dir", $OutputDir
)

Write-Host "Starting Training..." -ForegroundColor Green
Write-Host ""

# Execute training
$command = "python `"$($params -join '" "')`""
Invoke-Expression $command

Write-Host ""
Write-Host "Training Complete! Model saved at: $OutputDir" -ForegroundColor Green

Write-Host ""
Write-Host "Baseline: Val Acc=61.3%, no_rag=65.26%, naive_rag=46.51%" -ForegroundColor Yellow
