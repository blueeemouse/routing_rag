# 最终模型训练脚本
# 使用所有搜索得到的最优超参数
# 
# 最优参数来源:
#   - backbone: BAAI/bge-base-en-v1.5 (backbone搜索)
#   - learning_rate: 7e-5 (lr搜索 phaseB)
#   - class_weights: no_rag=6.8, naive_rag=1.0 (class_weights搜索)
#   - weight_decay: 10 (weight_decay搜索 phaseB)
#   - temperature: 0.5 (temperature搜索 bge)

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Green
Write-Host "最终模型训练 - BGE Backbone" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 最优超参数
$Backbone = "BAAI/bge-base-en-v1.5"
$LearningRate = "7e-5"
# $ClassWeights = "no_rag=6.8,naive_rag=1.0"
# $WeightDecay = "10"
$Temperature = "0.5"
$ClassWeights = "no_rag=1,naive_rag=1.5"
$WeightDecay = "0.01"

# 数据
# $TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_tie.json"
$TrainData = "D:/Develop/all_RAG/routing_rag/HotpotQA_train_data/10000/all_labels.json"

# 输出目录
$OutputDir = "router_models/model_bge_10000_naive_1.5"

Write-Host "配置:" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Cyan
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Cyan
Write-Host "  Class Weights: $ClassWeights" -ForegroundColor Cyan
Write-Host "  Weight Decay: $WeightDecay" -ForegroundColor Cyan
Write-Host "  Temperature: $Temperature" -ForegroundColor Cyan
Write-Host "  Train Data: $TrainData" -ForegroundColor Cyan
Write-Host "  Output Dir: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# 构建命令
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

Write-Host "启动训练..." -ForegroundColor Green
Write-Host ""

# 执行训练
$command = "python `"$($params -join '" "')`""
Invoke-Expression $command

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "训练完成!" -ForegroundColor Green
Write-Host "模型保存在: $OutputDir" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
