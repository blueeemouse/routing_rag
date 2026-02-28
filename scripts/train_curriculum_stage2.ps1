# Curriculum Learning Stage 2 Training Script
#
# Purpose: Continue training from Stage 1 best checkpoint with mixed data
#
# Curriculum Learning Design:
#   - Stage 1: Train on balanced non-tie samples (all_labels_balanced_1000.json)
#              Learn to distinguish when one strategy is clearly better
#   - Stage 2: Continue training on mixed samples (tie + non-tie)
#              Learn to apply efficiency prior for tie cases
#
# Stage 2 Dataset:
#   - Non-tie: 2000 samples (no_rag=1000, naive_rag=1000, balanced)
#   - Tie: 1000 samples (all labeled as no_rag - efficiency prior)
#   - Total: 3000 samples
#   - Ratio: Non-tie : Tie = 2 : 1
#
# Learning Rates to Compare:
#   - 7e-5: Same as Stage 1
#   - 7e-6: One order of magnitude smaller (more conservative)
#
# Usage:
#   .\scripts\train_curriculum_stage2.ps1              # Run both LR experiments
#   .\scripts\train_curriculum_stage2.ps1 -Stage1Checkpoint "path/to/checkpoint"

param(
    [string]$Stage1Checkpoint = "router_models/feature_fused_bge_20260228_002117/checkpoint_step_150"
)

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Curriculum Learning - Stage 2 Training (Two Learning Rates)" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Model config (same as Stage 1)
$ModelType = "feature_fused"
$Backbone = "BAAI/bge-base-en-v1.5"
$UseSpacy = $true
$FeatureNormalize = $true
$UseProjection = $true

# Stage 2 training hyperparameters
$WeightDecay = "10"
$Temperature = "0.5"
$BatchSize = 32
$Epochs = 5

# Class weights: no_rag=2000, naive_rag=1000 (2:1 ratio)
$ClassWeights = "no_rag=1.0,naive_rag=2.0"

# Data
$TrainData = "HotpotQA_train_data/label_analysis/curriculum_stage2/curriculum_stage2_mixed_3000.json"
$ValData = "evaluation_results/router_test_labels.json"

# Learning rates to compare
$LearningRates = @("7e-5", "7e-6")

Write-Host "Stage 1 Checkpoint: $Stage1Checkpoint" -ForegroundColor Yellow
Write-Host ""

Write-Host "Model Config:" -ForegroundColor Yellow
Write-Host "  Model Type: $ModelType" -ForegroundColor Cyan
Write-Host "  Backbone: $Backbone" -ForegroundColor Cyan
Write-Host ""

Write-Host "Training Config:" -ForegroundColor Yellow
Write-Host "  Learning Rates: $($LearningRates -join ', ')" -ForegroundColor Cyan
Write-Host "  Class Weights: $ClassWeights" -ForegroundColor Cyan
Write-Host "  Weight Decay: $WeightDecay" -ForegroundColor Cyan
Write-Host "  Batch Size: $BatchSize" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs" -ForegroundColor Cyan
Write-Host ""

Write-Host "Data:" -ForegroundColor Yellow
Write-Host "  Train: $TrainData" -ForegroundColor Cyan
Write-Host "  Val: $ValData" -ForegroundColor Cyan
Write-Host ""

# Check if Stage 1 checkpoint exists
if (-not (Test-Path $Stage1Checkpoint)) {
    Write-Host "ERROR: Stage 1 checkpoint not found: $Stage1Checkpoint" -ForegroundColor Red
    exit 1
}

# Train function
function Train-Model {
    param(
        [string]$LR,
        [string]$Checkpoint,
        [string]$OutDir
    )
    
    Write-Host "================================================================================" -ForegroundColor Magenta
    Write-Host "Training with Learning Rate: $LR" -ForegroundColor Magenta
    Write-Host "================================================================================" -ForegroundColor Magenta
    Write-Host ""
    
    $params = @(
        "router/train_router.py",
        "--config", "config/train_feature_fused.yaml",
        "--train_data", $TrainData,
        "--val_data", $ValData,
        "--model_type", $ModelType,
        "--backbone", $Backbone,
        "--learning_rate", $LR,
        "--temperature", $Temperature,
        "--class_weights", $ClassWeights,
        "--weight_decay", $WeightDecay,
        "--batch_size", $BatchSize.ToString(),
        "--epochs", $Epochs.ToString(),
        "--output_dir", $OutDir,
        "--resume", $Checkpoint
    )
    
    $command = "python `"$($params -join '" "')`""
    Invoke-Expression $command
    
    Write-Host ""
    Write-Host "Completed: $OutDir" -ForegroundColor Green
    Write-Host ""
}

# Run experiments
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Results = @()

foreach ($LR in $LearningRates) {
    # Create output directory name with LR info
    $LRName = $LR.Replace('-', '_')
    $OutputDir = "router_models/curriculum_stage2_lr_${LRName}_$Timestamp"
    
    Train-Model -LR $LR -Checkpoint $Stage1Checkpoint -OutDir $OutputDir
    
    $Results += @{LR=$LR; OutputDir=$OutputDir}
}

# Summary
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "All Stage 2 Experiments Complete!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Results:" -ForegroundColor Yellow
foreach ($r in $Results) {
    Write-Host "  LR=$($r.LR): $($r.OutputDir)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. View TensorBoard: tensorboard --logdir router_models/" -ForegroundColor Cyan
Write-Host "  2. Compare models: python scripts/compare_models.py --model1 <path1> --model2 <path2> --train_data $TrainData" -ForegroundColor Cyan
