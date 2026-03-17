# KNN Router Training Script
# KNN路由器训练脚本
#
# 【特点】基于K近邻的路由器，不需要梯度传播
#
# 训练过程：
# 1. 编码所有训练样本的embedding并存储
# 2. 可选：在验证集上搜索最优k值
# 3. 在验证集上评估性能
#
# 使用方法:
#   .\train_knn.ps1                    # 正常训练
#   .\train_knn.ps1 -QuickTest         # 快速测试（只用前100个样本）
#   .\train_knn.ps1 -K 7               # 指定k值
#   .\train_knn.ps1 -SearchK           # 搜索最优k值

param(
    [string]$Config = "config/train_knn.yaml",
    [string]$OutputDir = "",
    [int]$K = 5,
    [string]$Backbone = "BAAI/bge-base-en-v1.5",
    [int]$BatchSize = 64,
    [switch]$SearchK = $false,
    [switch]$QuickTest = $false,
    [int]$Seed = 42
)

# 时间戳
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 输出目录
if ($OutputDir -eq "") {
    $OutputDir = "router_models/knn_${timestamp}"
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "KNN Router 训练" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "配置文件: $Config"
Write-Host "输出目录: $OutputDir"
Write-Host "Backbone: $Backbone"
Write-Host "K值: $K"
Write-Host "Batch Size: $BatchSize"
Write-Host "搜索最优K: $SearchK"
Write-Host "Seed: $Seed"
Write-Host "=========================================" -ForegroundColor Cyan

# 构建命令
$command = "python router/train_router.py"
$command += " --config $Config"
$command += " --output_dir $OutputDir"
$command += " --backbone $Backbone"
$command += " --batch_size $BatchSize"
$command += " --seed $Seed"

# 快速测试模式
if ($QuickTest) {
    Write-Host "`n快速测试模式..." -ForegroundColor Yellow
    # 快速测试时使用较少的数据
    $command += " --train_data HotpotQA_train_data/10000/all_labels.json"
    $command += " --max_steps 10"
}

# 执行训练
Write-Host "`n开始训练..." -ForegroundColor Green
Write-Host "命令: $command" -ForegroundColor Gray
Invoke-Expression $command

Write-Host "`n训练完成!" -ForegroundColor Green
Write-Host "模型保存路径: $OutputDir" -ForegroundColor Green

# 可选：运行测试脚本
# Write-Host "`n运行测试..." -ForegroundColor Yellow
# python test_knn_router.py --model_path $OutputDir/final
