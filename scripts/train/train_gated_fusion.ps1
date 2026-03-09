# Gated-Fusion Router Training Script
#
# Model: Gated-Fusion Router (Semantic + Handcrafted Features with Gating)
#   P1方案：统计特征门控融合
#   - Semantic: BGE-base (768-dim)
#   - Handcrafted: 63-dim (based on RouteRAG paper)
#   - Gated Fusion: 自适应融合语义和统计特征
#   - 公式：fused = semantic * gate + stat_proj(stat) * (1 - gate)
#
# Data: Balanced 1000 samples (1:1 ratio)
#   - no_rag: 500
#   - naive_rag: 500
#
# Key features:
#   - Uses spaCy for dependency parsing and NER
#   - Gated mechanism for adaptive feature fusion
#   - Residual connection for better gradient flow

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Green
Write-Host "Gated-Fusion Router Training (P1)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Model config
$ModelType = "gated_fusion"
$Backbone = "BAAI/bge-base-en-v1.5"
$UseSpacy = $true
$FeatureNormalize = $true
$UseResidual = $true

# Training hyperparameters
$LearningRate = "7e-5"
$ClassWeights = "no_rag=1.0,naive_rag=1.0"  # Balanced data
$WeightDecay = "10"
$Temperature = "0.5"
$BatchSize = 32
$Epochs = 10

# Data
# $TrainData = "HotpotQA_train_data/label_analysis/balanced_samples/all_labels_balanced_1000.json"
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_rag_strictly_better_removed.json"
$ValData = "evaluation_results/router_test_labels.json"

# Output directory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = "router_models/gated_fusion_bge_$Timestamp"

Write-Host "Model Config:" -ForegroundColor Yellow
Write-Host "  Model Type: $ModelType (Gated Fusion)" -ForegroundColor Cyan
Write-Host "  Backbone: $Backbone" -ForegroundColor Cyan
Write-Host "  Use spaCy: $UseSpacy" -ForegroundColor Cyan
Write-Host "  Feature Normalize: $FeatureNormalize" -ForegroundColor Cyan
Write-Host "  Use Residual: $UseResidual" -ForegroundColor Cyan
Write-Host ""

Write-Host "Training Config:" -ForegroundColor Yellow
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Cyan
Write-Host "  Class Weights: $ClassWeights (balanced data)" -ForegroundColor Cyan
Write-Host "  Weight Decay: $WeightDecay" -ForegroundColor Cyan
Write-Host "  Temperature: $Temperature" -ForegroundColor Cyan
Write-Host "  Batch Size: $BatchSize" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs" -ForegroundColor Cyan
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
    "--config", "config/train_gated_fusion.yaml",
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
Write-Host "  3. Load model for inference: GatedFusionRouter('$OutputDir\final')" -ForegroundColor Cyan
