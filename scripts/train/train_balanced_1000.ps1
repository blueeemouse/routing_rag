# Balanced Dataset Training Script (1000 samples, 1:1 ratio)
# 
# Data characteristics:
#   - no_rag: 500 (361 original + 139 oversampled)
#   - naive_rag: 500 (undersampled)
#   - Ratio: 1:1 balanced
#
# Params transferred from previous search:
#   - backbone: BAAI/bge-base-en-v1.5
#   - learning_rate: 7e-5
#   - weight_decay: 10
#   - temperature: 0.5
#
# Adjusted:
#   - class_weights: 1:1 (data is balanced, no weighting needed)

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Green
Write-Host "Balanced Dataset Training (1000 samples, 1:1)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Hyperparameters (transferred from previous search, class_weights adjusted to 1:1)
$Backbone = "BAAI/bge-base-en-v1.5"
$LearningRate = "7e-5"
$ClassWeights = "no_rag=1.0,naive_rag=1.0"  # Balanced data, no weighting
$WeightDecay = "10"
$Temperature = "0.5"

# Data
$TrainData = "HotpotQA_train_data/label_analysis/balanced_samples/all_labels_balanced_1000.json"

# Output directory
$OutputDir = "router_models/balanced_1000_bge"

Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Cyan
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Cyan
Write-Host "  Class Weights: $ClassWeights (balanced data)" -ForegroundColor Cyan
Write-Host "  Weight Decay: $WeightDecay" -ForegroundColor Cyan
Write-Host "  Temperature: $Temperature" -ForegroundColor Cyan
Write-Host "  Train Data: $TrainData" -ForegroundColor Cyan
Write-Host "  Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Build command
$params = @(
    "router/train_router.py",
    "--config", "config/train_classification_5000.yaml",
    "--train_data", $TrainData,
    "--backbone", $Backbone,
    "--learning_rate", $LearningRate,
    "--temperature", $Temperature,
    "--class_weights", $ClassWeights,
    "--weight_decay", $WeightDecay,
    "--output_dir", $OutputDir
)

Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Execute training
$command = "python `"$($params -join '" "')`""
Invoke-Expression $command

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "Model saved at: $OutputDir" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
