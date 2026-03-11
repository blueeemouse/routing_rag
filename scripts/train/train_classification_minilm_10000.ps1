# Classification Router Training Script (MiniLM)
#
# Model: Classification Router (Direct Classification)
#   - Backbone: MiniLM-L6-v2 (384-dim)
#   - Loss: Cross Entropy
#   - Training: No-RAG vs Naive-RAG classification
#
# Data: Full 10000 samples
#   - no_rag: 750
#   - naive_rag: 3908
#   - tie: 5342 (will be used as training data)
#
# Key features:
#   - Simple classification head
#   - Efficient training with cross entropy loss
#   - Suitable for balanced/imbalanced data

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Green
Write-Host "Classification Router Training (MiniLM)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Model config
$ModelType = "dc"
$Backbone = "sentence-transformers/all-MiniLM-L6-v2"
$Config = "config/train_classification_5000.yaml"

# Training hyperparameters (can override config values)
$LearningRate = "2e-4"  # Config default: 0.0002
$BatchSize = 32         # Config default: 32
$Epochs = 10            # Config default: 10
$Temperature = "0.5"    # Config default: 0.5
$ClassWeights = "no_rag=1.0,naive_rag=1.8"  # 提高 naive_rag 权重

# Data
$TrainData = "D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/10000/all_labels.json"
$ValData = "evaluation_results/router_test_labels.json"

# Output directory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = "router_models/classification_minilm_10000_$Timestamp"

Write-Host "Model Config:" -ForegroundColor Yellow
Write-Host "  Model Type: $ModelType (Direct Classification)" -ForegroundColor Cyan
Write-Host "  Backbone: $Backbone" -ForegroundColor Cyan
Write-Host "  Config: $Config" -ForegroundColor Cyan
Write-Host ""

Write-Host "Training Config:" -ForegroundColor Yellow
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Cyan
Write-Host "  Batch Size: $BatchSize" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs" -ForegroundColor Cyan
Write-Host "  Temperature: $Temperature" -ForegroundColor Cyan
Write-Host "  Class Weights: $ClassWeights" -ForegroundColor Cyan
Write-Host ""

Write-Host "Data:" -ForegroundColor Yellow
Write-Host "  Train Data: $TrainData" -ForegroundColor Cyan
Write-Host "  Val Data: $ValData" -ForegroundColor Cyan
Write-Host ""

Write-Host "Output:" -ForegroundColor Yellow
Write-Host "  Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Build command
$params = @(
    "router/train_router.py",
    "--config", $Config,
    "--train_data", $TrainData,
    "--val_data", $ValData,
    "--model_type", $ModelType,
    "--backbone", $Backbone,
    "--learning_rate", $LearningRate,
    "--temperature", $Temperature,
    "--class_weights", $ClassWeights,
    "--batch_size", $BatchSize.ToString(),
    "--epochs", $Epochs.ToString(),
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

# Print next steps
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Check training logs: $OutputDir\logs\" -ForegroundColor Cyan
Write-Host "  2. View TensorBoard: tensorboard --logdir $OutputDir\tensorboard\" -ForegroundColor Cyan
Write-Host "  3. Load model for inference: DCRouter('$OutputDir\final')" -ForegroundColor Cyan
