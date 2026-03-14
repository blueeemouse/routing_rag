# Binary Soft Label Router Training Script
# 二分类软标签路由器训练脚本
#
# 【重要】此脚本仅支持二分类任务（no_rag vs naive_rag）
#
# 训练配置：
# - 模型: StatisticalRouterModel (纯统计特征, 63维手工特征)
# - 训练器: BinarySoftLabelTrainer (BCEWithLogitsLoss)
# - 数据集: BinarySoftLabelDataset (单值软标签)
# - 数据: all_labels_soft.json (包含 soft_label 字段)
# - 软标签公式: sigmoid((Q_naive_rag - Q_no_rag) / τ), τ=0.1
#
# 使用方法:
#   .\train_soft_label.ps1                    # 正常训练
#   .\train_soft_label.ps1 -QuickTest         # 快速测试 (10 steps)
#   .\train_soft_label.ps1 -Epochs 20         # 自定义 epochs

param(
    [string]$Config = "config/train_soft_label.yaml",
    [string]$OutputDir = "",
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [double]$LearningRate = 0.0001,
    [double]$Threshold = 0.5,
    [int]$EvalSteps = 50,
    [int]$Seed = 42,
    [switch]$QuickTest = $false,
    [string]$Resume = ""
)

# 时间戳
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 输出目录
if ($OutputDir -eq "") {
    $OutputDir = "router_models/soft_label_${timestamp}"
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "软标签路由器训练" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "配置文件: $Config"
Write-Host "输出目录: $OutputDir"
Write-Host "Epochs: $Epochs"
Write-Host "Batch Size: $BatchSize"
Write-Host "Learning Rate: $LearningRate"
Write-Host "Threshold: $Threshold"
Write-Host "Eval Steps: $EvalSteps"
Write-Host "Seed: $Seed"
Write-Host "=========================================" -ForegroundColor Cyan

# 构建命令
$command = "python router/train_router.py"
$command += " --config $Config"
$command += " --output_dir $OutputDir"
$command += " --epochs $Epochs"
$command += " --batch_size $BatchSize"
$command += " --learning_rate $LearningRate"
$command += " --eval_steps $EvalSteps"
$command += " --seed $Seed"

if ($Resume -ne "") {
    $command += " --resume $Resume"
}

# 快速测试模式
if ($QuickTest) {
    $command += " --max_steps 10 --overfit_single_batch"
    Write-Host "`n快速测试模式 (10 steps)..." -ForegroundColor Yellow
} else {
    Write-Host "`n开始训练..." -ForegroundColor Green
}

# 执行训练
Write-Host "命令: $command" -ForegroundColor Gray
Invoke-Expression $command

Write-Host "`n训练完成!" -ForegroundColor Green
Write-Host "模型保存路径: $OutputDir" -ForegroundColor Green
