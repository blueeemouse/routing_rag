# Weight Decay Search - Balanced Dataset
# 
# Background: With balanced data (1:1), overfitting risk is reduced
# Purpose: Confirm if large weight_decay is still needed
#
# Fixed params:
#   - backbone: BAAI/bge-base-en-v1.5
#   - lr: 7e-5
#   - temperature: 0.5
#   - class_weights: 1:1 (balanced data)
#
# Search params:
#   - weight_decay: [0.01, 0.1, 1.0, 5.0, 10.0]

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Fixed parameters
$Backbone = "BAAI/bge-base-en-v1.5"
$LearningRate = "7e-5"
$Temperature = "0.5"
$ClassWeights = "no_rag=1.0,naive_rag=1.0"
$TrainData = "HotpotQA_train_data/label_analysis/balanced_samples/all_labels_balanced_1000.json"

# Search range
$WeightDecays = @("0.01", "0.1", "1.0", "5.0", "10.0")

# Output directory
$ExperimentsRoot = "router_models/wd_search_balanced"
$LogFile = "$ExperimentsRoot/wd_search_log.txt"

# Helper Functions
function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

# Main
Write-Host "========================================" -ForegroundColor Green
Write-Host "Weight Decay Search - Balanced Dataset" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Data: $TrainData (1000 samples, 1:1 balanced)" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Yellow
Write-Host "  Learning Rate: $LearningRate (fixed)" -ForegroundColor Yellow
Write-Host "  Temperature: $Temperature (fixed)" -ForegroundColor Yellow
Write-Host "  Class Weights: $ClassWeights (balanced data)" -ForegroundColor Yellow
Write-Host "  Weight Decay: $($WeightDecays -join ', ')" -ForegroundColor Yellow
Write-Host "  Output Dir: $ExperimentsRoot" -ForegroundColor Yellow
Write-Host ""

# Create output directory
if (-not (Test-Path $ExperimentsRoot)) {
    New-Item -ItemType Directory -Path $ExperimentsRoot -Force | Out-Null
}

# Clear log file
if (Test-Path $LogFile) {
    Clear-Content $LogFile
}

Log-Message "Weight Decay Search Config:"
Log-Message "  Data: $TrainData"
Log-Message "  Backbone: $Backbone"
Log-Message "  Learning Rate: $LearningRate"
Log-Message "  Temperature: $Temperature"
Log-Message "  Class Weights: $ClassWeights"
Log-Message "  Weight Decays: $($WeightDecays -join ', ')"
Log-Message "  Output Dir: $ExperimentsRoot"
Log-Message ""

# Statistics
$TotalExperiments = $WeightDecays.Count
$CompletedExperiments = 0
$FailedExperiments = 0
$StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow
Write-Host ""

# Experiment loop
$ExperimentIndex = 0
foreach ($wd in $WeightDecays) {
    $ExperimentIndex++
    
    # Output directory name
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
    
    # Show current experiment info
    $progress = "[{0}/{1}] Weight Decay: {2}" -f $ExperimentIndex, $TotalExperiments, $wd
    Write-Host $progress -ForegroundColor Cyan
    Write-Host "  Output: $output_dir" -ForegroundColor Gray
    
    # Log execution
    Log-Message "Starting: WD=$wd"
    Log-Message "  Command: python $($params -join ' ')"
    
    # Execute training
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
    
    # Check results
    if ($exitCode -eq 0) {
        $CompletedExperiments++
        Log-Message "  [OK] Experiment Finished: WD=$wd"
        Write-Host "  [OK] Finished" -ForegroundColor Green
    }
    else {
        $FailedExperiments++
        Log-Message "  [FAIL] Experiment Failed: WD=$wd (Exit Code: $exitCode)"
        Write-Host "  [FAIL] Failed (Exit Code: $exitCode)" -ForegroundColor Red
    }
    
    Log-Message ""
    Write-Host ""
}

# Summary
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "========================================" -ForegroundColor Green
Write-Host "Weight Decay Search Complete" -ForegroundColor Green
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
