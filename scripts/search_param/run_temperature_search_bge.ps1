# Temperature Search for BGE Backbone
# 固定参数：backbone=BAAI/bge-base-en-v1.5, lr=7e-5, class_weights=6.8:1, weight_decay=10

$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 固定参数
$Backbone = "BAAI/bge-base-en-v1.5"
$LearningRate = "7e-5"
$ClassWeights = "no_rag=6.8,naive_rag=1.0"
$WeightDecay = "10"
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json"

# Temperature 搜索范围
$Temperatures = @(0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)

# 输出目录
$ExperimentsRoot = "router_models/temperature_search_bge"
$LogFile = "$ExperimentsRoot/temperature_search_log.txt"

# 创建目录
if (-not (Test-Path $ExperimentsRoot)) {
    New-Item -ItemType Directory -Path $ExperimentsRoot -Force | Out-Null
}

# 日志函数
function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] $Message"
    Write-Host $LogEntry
    Add-Content -Path $LogFile -Value $LogEntry
}

Write-Log "========================================"
Write-Log "Temperature Search for BGE Backbone"
Write-Log "========================================"
Write-Log ""
Write-Log "Fixed Hyperparameters:"
Write-Log "  Backbone: $Backbone"
Write-Log "  Learning Rate: $LearningRate"
Write-Log "  Class Weights: $ClassWeights"
Write-Log "  Weight Decay: $WeightDecay"
Write-Log "  Train Data: $TrainData"
Write-Log ""
Write-Log "Temperature Range: $($Temperatures -join ', ')"
Write-Log ""

$StartTime = Get-Date
$TotalExperiments = $Temperatures.Count
$SuccessCount = 0
$FailCount = 0

for ($i = 0; $i -lt $TotalExperiments; $i++) {
    $Temp = $Temperatures[$i]
    $Progress = $i + 1
    
    Write-Log "[$Progress/$TotalExperiments] Temperature: $Temp"
    Write-Log "----------------------------------------"
    
    $OutputDir = "$ExperimentsRoot/temp_$Temp"
    
    # 运行训练
    $Command = "python router/train_router.py --config config/train_classification_5000.yaml --train_data `"$TrainData`" --backbone `"$Backbone`" --learning_rate $LearningRate --temperature $Temp --class_weights `"$ClassWeights`" --weight_decay $WeightDecay --output_dir `"$OutputDir`""
    
    Write-Log "Command: $Command"
    
    $ExpStart = Get-Date
    Invoke-Expression $Command 2>&1 | Tee-Object -FilePath "$OutputDir/training_output.log" | Out-Null
    $ExpEnd = Get-Date
    $ExpDuration = $ExpEnd - $ExpStart
    
    if ($LASTEXITCODE -eq 0) {
        Write-Log "  [OK] Completed in $($ExpDuration.TotalMinutes.ToString('F1')) min"
        $SuccessCount++
    } else {
        Write-Log "  [FAIL] Experiment Failed: temperature=$Temp (Exit Code: $LASTEXITCODE)"
        $FailCount++
    }
    Write-Log ""
}

$EndTime = Get-Date
$TotalDuration = $EndTime - $StartTime

Write-Log "========================================"
Write-Log "Temperature Search Complete"
Write-Log "========================================"
Write-Log "Total: $TotalExperiments, Success: $SuccessCount, Failed: $FailCount"
Write-Log "Total Time: $($TotalDuration.TotalMinutes.ToString('F1')) min"
Write-Log ""
Write-Log "Next Step: Compare validation accuracy to find optimal temperature"
Write-Log "Output Dir: $ExperimentsRoot"
