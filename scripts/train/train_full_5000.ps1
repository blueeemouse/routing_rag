# Full 5000 Samples Training Script
#
# Purpose: Train router model with all 5000 samples (no curriculum learning)
#
# Data:
#   - Source: all_labels.json (converted to RouterLabelDataset format)
#   - tie samples converted to no_rag (efficiency prior)
#   - Distribution: no_rag=2910, naive_rag=2090
#
# Key Settings:
#   - Backbone: BAAI/bge-base-en-v1.5
#   - Learning Rate: 7e-5
#   - Class Weights: 1:1 (natural distribution)
#   - No curriculum learning
#
# Usage:
#   .\scripts\train_full_5000.ps1

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Full 5000 Samples Training" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Model config
$ModelType = "feature_fused"
$Backbone = "BAAI/bge-base-en-v1.5"

# Training hyperparameters
$LearningRate = "7e-5"
$WeightDecay = "10"
$Temperature = "0.5"
$BatchSize = 32
$Epochs = 10

# Class weights: 1:1 (natural distribution, no balancing)
$ClassWeights = "no_rag=1.0,naive_rag=1.0"

# Data
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json"
$ValData = "evaluation_results/router_test_labels.json"

# Output directory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = "router_models/full_5000_bge_$Timestamp"

Write-Host "Model Config:" -ForegroundColor Yellow
Write-Host "  Model Type: $ModelType" -ForegroundColor Cyan
Write-Host "  Backbone: $Backbone" -ForegroundColor Cyan
Write-Host ""

Write-Host "Training Config:" -ForegroundColor Yellow
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Cyan
Write-Host "  Weight Decay: $WeightDecay" -ForegroundColor Cyan
Write-Host "  Temperature: $Temperature" -ForegroundColor Cyan
Write-Host "  Batch Size: $BatchSize" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs" -ForegroundColor Cyan
Write-Host "  Class Weights: $ClassWeights (natural distribution)" -ForegroundColor Cyan
Write-Host ""

Write-Host "Data:" -ForegroundColor Yellow
Write-Host "  Train: $TrainData" -ForegroundColor Cyan
Write-Host "    - no_rag: 2910 (including 2549 tie samples)" -ForegroundColor Cyan
Write-Host "    - naive_rag: 2090" -ForegroundColor Cyan
Write-Host "    - Total: 5000" -ForegroundColor Cyan
Write-Host "  Val: $ValData" -ForegroundColor Cyan
Write-Host ""

Write-Host "Output:" -ForegroundColor Yellow
Write-Host "  Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Build command
$params = @(
    "router/train_router.py",
    "--config", "config/train_feature_fused.yaml",
    "--train_data", $TrainData,
    "--val_data", $ValData,
    "--model_type", $ModelType,
    "--backbone", $Backbone,
    "--learning_rate", $LearningRate,
    "--temperature", $Temperature,
    "--class_weights", $ClassWeights,
    "--weight_decay", $WeightDecay,
    "--batch_size", $BatchSize.ToString(),
    "--epochs", $Epochs.ToString(),
    "--output_dir", $OutputDir
)

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Starting Training..." -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""

# Execute training
$command = "python `"$($params -join '" "')`""
Invoke-Expression $command

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "Model saved at: $OutputDir" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Green

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. View TensorBoard: tensorboard --logdir $OutputDir\tensorboard\" -ForegroundColor Cyan
Write-Host "  2. Compare with other models using compare_models.py" -ForegroundColor Cyan
