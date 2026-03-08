# 搜索一下更大的weight_decay值（因为之前搜索的感觉过小，导致它们的结果相差无几）
# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Weight Decay Search - Phase B: Larger Values
# Phase A showed minimal effect, trying larger values to observe impact

# Fixed Hyperparameters (from previous search)
$Backbone = "BAAI/bge-base-en-v1.5"
$LearningRate = "7e-5"
$Temperature = 0.5
$ClassWeights = "no_rag=6.8,naive_rag=1.0"  # inverse_freq (best from previous search)
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json"

# Weight decay values to test (larger scale)
$WeightDecayValues = @("1.0", "3.0", "5.0", "10.0", "20.0")

# Output Directory
$ExperimentsRoot = "router_models/weight_decay_search_phaseB"
$LogFile = "$ExperimentsRoot/weight_decay_search_log.txt"

# Helper Functions
function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

# Main Execution
Write-Host "========================================" -ForegroundColor Green
Write-Host "Weight Decay Search - Phase B (Larger Values)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Phase A Result: Minimal effect observed with 0.0 ~ 1.0" -ForegroundColor Yellow
Write-Host ""
Write-Host "Fixed Config:" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Yellow
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Yellow
Write-Host "  Temperature: $Temperature" -ForegroundColor Yellow
Write-Host "  Class Weights: $ClassWeights" -ForegroundColor Yellow
Write-Host "  Train Data: $TrainData" -ForegroundColor Yellow
Write-Host ""
Write-Host "Weight Decay to Test: $($WeightDecayValues -join ', ')" -ForegroundColor Yellow
Write-Host ""

# Create Output Directory
if (-not (Test-Path $ExperimentsRoot)) {
    New-Item -ItemType Directory -Path $ExperimentsRoot -Force | Out-Null
}

# Clear Log File
if (Test-Path $LogFile) {
    Clear-Content $LogFile
}

Log-Message "Weight Decay Search Phase B Config:"
Log-Message "  Backbone: $Backbone"
Log-Message "  Learning Rate: $LearningRate"
Log-Message "  Temperature: $Temperature"
Log-Message "  Class Weights: $ClassWeights"
Log-Message "  Train Data: $TrainData"
Log-Message "  Weight Decay values: $($WeightDecayValues -join ', ')"
Log-Message ""

# Init Statistics
$TotalExperiments = $WeightDecayValues.Count
$CompletedExperiments = 0
$FailedExperiments = 0
$StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow
Write-Host ""

# Experiment Loop
$ExperimentIndex = 0
foreach ($wd in $WeightDecayValues) {
    $ExperimentIndex++
    
    # Generate Output Directory Name
    $output_dir = "$ExperimentsRoot/wd_$wd"
    
    # Build parameters
    $params = @(
        "router/train_router.py"
        "--config", "config/train_classification_5000.yaml"
        "--train_data", $TrainData
        "--backbone", $Backbone
        "--learning_rate", $LearningRate
        "--temperature", $Temperature
        "--class_weights", $ClassWeights
        "--weight_decay", $wd
        "--output_dir", $output_dir
    )
    
    # Show Current Experiment Info
    $progress = "[{0}/{1}] Weight Decay: {2}" -f $ExperimentIndex, $TotalExperiments, $wd
    Write-Host $progress -ForegroundColor Cyan
    Write-Host "  Output: $output_dir" -ForegroundColor Gray
    
    # Log Execution
    Log-Message "Starting: weight_decay=$wd"
    Log-Message "  Command: python $($params -join ' ')"
    
    # Execute Training
    $exitCode = 0
    
    try {
        $command = "python `"$($params -join '" "')`""
        Invoke-Expression $command
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq $null) {
            $exitCode = 0
        }
    }
    catch {
        $exitCode = -1
        Log-Message "  [ERROR] Execution Failed: $_"
    }
    
    # Check Results
    if ($exitCode -eq 0) {
        $CompletedExperiments++
        Log-Message "  [OK] Experiment Finished: weight_decay=$wd"
        Write-Host "  [OK] Finished" -ForegroundColor Green
    }
    else {
        $FailedExperiments++
        Log-Message "  [FAIL] Experiment Failed: weight_decay=$wd (Exit Code: $exitCode)"
        Write-Host "  [FAIL] Failed (Exit Code: $exitCode)" -ForegroundColor Red
    }
    
    Log-Message ""
    Write-Host ""
}

# Summary
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "========================================" -ForegroundColor Green
Write-Host "Weight Decay Search Phase B Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Log-Message "Summary:"
Log-Message "  Total Experiments: $TotalExperiments"
Log-Message "  Success: $CompletedExperiments"
Log-Message "  Failed: $FailedExperiments"
Log-Message "  Start Time: $StartTime"
Log-Message "  End Time: $EndTime"
Log-Message "  Duration: $Duration"
Log-Message ""
Log-Message "Results saved in: $ExperimentsRoot"

Write-Host ""
Write-Host "Experiment Stats: Success=$CompletedExperiments/$TotalExperiments" -ForegroundColor $(if ($CompletedExperiments -eq $TotalExperiments) { "Green" } else { "Yellow" })
Write-Host "Duration: $Duration" -ForegroundColor Yellow
Write-Host ""
Write-Host "Results saved in: $ExperimentsRoot" -ForegroundColor Yellow
Write-Host "Log file: $LogFile" -ForegroundColor Yellow
Write-Host ""
