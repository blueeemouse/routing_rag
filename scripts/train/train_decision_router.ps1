# Decision Router Training Script
# 决策式路由器训练脚本
#
# 用法：
# .\train_decision_router.ps1 [-Config "config\train_decision_router.yaml"] [-Overfit]
#
# 参数：
# -Config: 配置文件路径（默认使用 config\train_decision_router.yaml）
# -Overfit: 使用过拟合模式进行调试

param(
    [string]$Config = "config\train_decision_router.yaml",
    [switch]$Overfit = $false,
    [int]$MaxSteps = 0,
    [string]$Resume = ""
)

# 激活虚拟环境（如果需要）
# & .\source_env_var.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Decision Router Training" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 构建命令
$cmd = "python router\train_router.py --config `$Config`""

if ($Overfit) {
    $cmd += " --overfit_single_batch --fast_dev_steps 50"
    Write-Host "[Debug Mode] Overfitting single batch for 50 steps" -ForegroundColor Yellow
}

if ($MaxSteps -gt 0) {
    $cmd += " --max_steps $MaxSteps"
    Write-Host "[Debug Mode] Max steps: $MaxSteps" -ForegroundColor Yellow
}

if ($Resume -ne "") {
    $cmd += " --resume `$Resume`""
    Write-Host "Resuming from: $Resume" -ForegroundColor Green
}

Write-Host "Command: $cmd" -ForegroundColor Gray
Write-Host ""

# 执行训练
Invoke-Expression $cmd

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
