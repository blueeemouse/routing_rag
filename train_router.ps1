# Trainable Router 训练脚本 (Windows PowerShell)

# 设置参数
param(
    [string]$TrainData = "evaluation_results",
    [string]$OutputDir = "router_models",
    [string]$ModelType = "dc",
    [string]$Backbone = "sentence-transformers/all-MiniLM-L6-v2",
    [int]$BatchSize = 32,
    [double]$LearningRate = 5.0e-5,
    [int]$Epochs = 10,
    [int]$MaxLength = 512,
    [int]$EvalSteps = 100,
    [int]$SaveSteps = 500,
    [int]$Seed = 42,
    [string]$Config = "",
    [string]$Resume = "",
    [switch]$QuickTest = $false  # 快速测试模式（只跑几个iter）
)

# 显示参数
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "路由器训练脚本" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "训练数据: $TrainData"
Write-Host "输出目录: $OutputDir"
Write-Host "模型类型: $ModelType"
Write-Host "Backbone: $Backbone"
Write-Host "批量大小: $BatchSize"
Write-Host "学习率: $LearningRate"
Write-Host "训练轮数: $Epochs"
Write-Host "=========================================" -ForegroundColor Cyan

# 构建命令
$command = "python router/train_router.py"
$command += " --train_data $TrainData"
$command += " --output_dir $OutputDir"
$command += " --model_type $ModelType"
$command += " --backbone `"$Backbone`""
$command += " --batch_size $BatchSize"
$command += " --learning_rate $LearningRate"
$command += " --epochs $Epochs"
$command += " --max_length $MaxLength"
$command += " --eval_steps $EvalSteps"
$command += " --save_steps $SaveSteps"
$command += " --seed $Seed"

if ($Config -ne "") {
    $command += " --config $Config"
}

if ($Resume -ne "") {
    $command += " --resume $Resume"
}

# 添加脚本路径参数（用于记录脚本内容到日志）
$scriptPath = $MyInvocation.MyCommand.Path
$command += " --script_path `"$scriptPath`""

# 快速测试模式
if ($QuickTest) {
    $command += " --max_steps 10"  # 只跑10个step
    Write-Host "`n快速测试模式（10 steps）..." -ForegroundColor Yellow
} else {
    Write-Host "`n开始训练..." -ForegroundColor Green
}

# 执行训练
Invoke-Expression $command

Write-Host "`n训练完成!" -ForegroundColor Green
Write-Host "模型保存路径: $OutputDir/final" -ForegroundColor Green

# 评估模型
Write-Host "`n评估模型..." -ForegroundColor Green
$evalCommand = "python router/trainable_router/evaluate_router.py"
$evalCommand += " --model_path $OutputDir/final"
$evalCommand += " --test_data $TrainData"
Invoke-Expression $evalCommand
