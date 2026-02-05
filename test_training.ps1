# 测试训练脚本 - 快速验证日志记录

# 设置参数
$TrainData = "evaluation_results"
$OutputDir = "router_models/test_no_rag_vs_naive"
$Config = "config/train_no_rag_vs_naive.yaml"

# 显示参数
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "测试Router训练" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "训练数据: $TrainData"
Write-Host "输出目录: $OutputDir"
Write-Host "配置文件: $Config"
Write-Host "=========================================" -ForegroundColor Cyan

# 构建命令
$command = "python router/train_router.py"
$command += " --config $Config"
$command += " --output_dir $OutputDir"

# 添加脚本路径参数
$scriptPath = $MyInvocation.MyCommand.Path
$command += " --script_path `"$scriptPath`""

# 快速测试模式（只跑几个iter）
$command += " --max_steps 5"  # 只跑5个step

Write-Host "`n快速测试模式（5 steps）..." -ForegroundColor Yellow
Write-Host "命令: $command" -ForegroundColor Gray

# 执行训练
Invoke-Expression $command

Write-Host "`n测试完成!" -ForegroundColor Green
