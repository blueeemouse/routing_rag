# 超参数网格搜索脚本
# 用于批量运行不同超参数组合的训练实验

# ========== 参数定义 ==========

# 温度参数列表
$Temperatures = @(0.05, 0.1, 0.2, 0.5, 1.0)

# 类别权重配置列表
# 格式：@("no_rag_weight", "描述")
$WeightConfigs = @(
    @("1.0", "无加权（基准）"),
    @("1.5", "轻微加权"),
    @("2.0", "中等加权"),
    @("3.0", "较强加权"),
    @("5.0", "强加权")
)

# Backbone选择
$Backbone = "all-MiniLM-L6-v2"
# $Backbone = "bert-base-uncased"  # 可选

# ========== 基础参数 ==========

$BaseArgs = @(
    "--model_type", "classification",
    "--config", "config/train_classification_5000.yaml",
    "--train_data", "HotpotQA_train_data",
    "--val_data", "evaluation_results/router_test_labels.json",
    "--overfit_single_batch",
    "--fast_dev_steps", "10"
    # 删除下面这一行进行全量训练：
    # "--overfit_single_batch", "--fast_dev_steps", "10"
)

# ========== 输出目录 ==========

$ExperimentsRoot = "router_models/grid_search"
$LogFile = "$ExperimentsRoot/grid_search_log.txt"

# ========== 辅助函数 ==========

function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

function Get-Experiment-Name {
    param(
        [double]$temp,
        [string]$weight_desc,
        [string]$backbone
    )
    # 生成实验名称：exp_<temp>_w<weight>_<backbone>_classification_<timestamp>
    $timestamp = Get-Date -Format "MMdd_HHmm"
    $exp_name = "exp_t${temp}_w${weight_desc}_$(${backbone}_classification)_$timestamp"
    return $exp_name
}

# ========== 主程序 ==========

Write-Host "========================================" -ForegroundColor Green
Write-Host "开始超参数网格搜索" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 创建输出目录
if (-not (Test-Path $ExperimentsRoot)) {
    New-Item -ItemType Directory -Path $ExperimentsRoot -Force | Out-Null
}

# 清空日志文件
if (Test-Path $LogFile) {
    Clear-Content $LogFile
}

Log-Message "网格搜索配置："
Log-Message "  温度列表: $($Temperatures -join ', ')"
Log-Message "  权重配置: $($WeightConfigs | ForEach-Object { $_.Split(',')[0] } | Join-Object -Separator ', ')"
Log-Message "  Backbone: $Backbone"
Log-Message "  输出目录: $ExperimentsRoot"
Log-Message "  日志文件: $LogFile"
Log-Message ""

# 初始化统计
$TotalExperiments = $Temperatures.Count * $WeightConfigs.Count
$CompletedExperiments = 0
$FailedExperiments = 0
$StartTime = Get-Date

Write-Host "准备运行 $TotalExperiments 个实验..." -ForegroundColor Yellow
Write-Host ""

# ========== 实验循环 ==========

foreach ($temp in $Temperatures) {
    foreach ($weightConfig in $WeightConfigs) {
        $parts = $weightConfig -split ','
        $no_rag_weight = $parts[0]
        $weight_desc = $parts[1].Trim()
        
        # 构造权重字符串
        $WeightStr = "no_rag=$no_rag_weight,naive_rag=1.0"
        
        # 生成实验名称
        $exp_name = Get-Experiment-Name -temp $temp -weight_desc $weight_desc -backbone $Backbone
        $output_dir = "$ExperimentsRoot/$exp_name"
        
        # 构造完整命令
        $command = "python router/train_router.py $($BaseArgs -join ' ') --backbone $Backbone --temperature $temp --class_weights `"$WeightStr`" --output_dir `"$output_dir`""
        
        # 显示当前实验信息
        $progress = "[{0}/{1} Temp={2}, Weight={3}" -f ($CompletedExperiments + 1), $TotalExperiments, $temp, $weight_desc
        Write-Host $progress -ForegroundColor Cyan
        
        # 记录日志
        Log-Message "开始实验: $exp_name"
        Log-Message "  命令: $command"
        
        # 执行训练
        $exitCode = 0
        try {
            & python $command.Split(' ')
            $exitCode = $LASTEXITCODE
        }
        catch {
            $exitCode = $_.Exception.HResult
            Log-Message "  【错误】执行失败: $_"
        }
        
        # 检查结果
        if ($exitCode -eq 0) {
            $CompletedExperiments++
            Log-Message "  ✓ 实验完成: $exp_name"
        }
        else {
            $FailedExperiments++
            Log-Message "  ✗ 实验失败: $exp_name (退出码: $exitCode)"
        }
        
        Log-Message ""
    }
}

# ========== 汇总 ==========

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "网格搜索完成" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Log-Message "总结："
Log-Message "  总实验数: $TotalExperiments"
Log-Message "  成功: $CompletedExperiments"
Log-Message "  失败: $FailedExperiments"
Log-Message "  开始时间: $StartTime"
Log-Message "  结束时间: $EndTime"
Log-Message "  总耗时: $Duration"
Log-Message ""
Log-Message "详细日志请查看: $LogFile"

Write-Host ""
Write-Host "实验统计：成功=$CompletedExperiments/$TotalExperiments" -ForegroundColor $(if ($CompletedExperiments -eq $TotalExperiments) { "Green" } else { "Yellow" })
Write-Host ""
