# 这个文件是搜索类别权重的。注意和之前的run_weight_search.ps1不同，这里的骨干换成更大的了，而且lr针对性也调过了
# 而且这里的搜索是为了快速训练，采用了采样训练数据的
# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Class Weights Search
# Based on data distribution: no_rag=147, naive_rag=853 (ratio ~1:5.8)
# Testing different weighting strategies

# Fixed Hyperparameters (from previous search)
$Backbone = "BAAI/bge-base-en-v1.5"
$LearningRate = "7e-5"
$Temperature = 0.5
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json"

# Class weights to test
# Format: "no_rag=X,naive_rag=Y"
$ClassWeightsConfigs = @(
    @{
        Name = "no_weight"
        Value = "no_rag=1.0,naive_rag=1.0"
        Note = "No weighting (1:1)"
    },
    @{
        Name = "sqrt_inverse"
        Value = "no_rag=2.6,naive_rag=1.1"
        Note = "Sqrt inverse frequency (current)"
    },
    @{
        Name = "weight_3to1"
        Value = "no_rag=3.0,naive_rag=1.0"
        Note = "Moderate weighting (3:1)"
    },
    @{
        Name = "weight_4to1"
        Value = "no_rag=4.0,naive_rag=1.0"
        Note = "Moderate weighting (4:1)"
    },
    @{
        Name = "weight_5to1"
        Value = "no_rag=5.0,naive_rag=1.0"
        Note = "Moderate weighting (5:1)"
    },
    @{
        Name = "inverse_freq"
        Value = "no_rag=6.8,naive_rag=1.0"
        Note = "Full inverse frequency"
    }
)

# Output Directory
$ExperimentsRoot = "router_models/class_weights_search"
$LogFile = "$ExperimentsRoot/class_weights_search_log.txt"

# Helper Functions
function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

# Main Execution
Write-Host "========================================" -ForegroundColor Green
Write-Host "Class Weights Search" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Data Distribution:" -ForegroundColor Yellow
Write-Host "  no_rag: ~147 (14.7%)" -ForegroundColor Gray
Write-Host "  naive_rag: ~853 (85.3%)" -ForegroundColor Gray
Write-Host ""
Write-Host "Fixed Config:" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Yellow
Write-Host "  Learning Rate: $LearningRate" -ForegroundColor Yellow
Write-Host "  Temperature: $Temperature" -ForegroundColor Yellow
Write-Host "  Train Data: $TrainData" -ForegroundColor Yellow
Write-Host ""
Write-Host "Weights to Test:" -ForegroundColor Yellow
foreach ($config in $ClassWeightsConfigs) {
    Write-Host "  - $($config.Name): $($config.Value) ($($config.Note))" -ForegroundColor Gray
}
Write-Host ""

# Create Output Directory
if (-not (Test-Path $ExperimentsRoot)) {
    New-Item -ItemType Directory -Path $ExperimentsRoot -Force | Out-Null
}

# Clear Log File
if (Test-Path $LogFile) {
    Clear-Content $LogFile
}

Log-Message "Class Weights Search Config:"
Log-Message "  Backbone: $Backbone"
Log-Message "  Learning Rate: $LearningRate"
Log-Message "  Temperature: $Temperature"
Log-Message "  Train Data: $TrainData"
Log-Message "  Weights configs: $($ClassWeightsConfigs.Count)"
Log-Message ""

# Init Statistics
$TotalExperiments = $ClassWeightsConfigs.Count
$CompletedExperiments = 0
$FailedExperiments = 0
$StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow
Write-Host ""

# Experiment Loop
$ExperimentIndex = 0
foreach ($config in $ClassWeightsConfigs) {
    $ExperimentIndex++
    
    $configName = $config.Name
    $classWeights = $config.Value
    
    # Generate Output Directory Name
    $output_dir = "$ExperimentsRoot/$configName"
    
    # Build parameters
    $params = @(
        "router/train_router.py"
        "--config", "config/train_classification_5000.yaml"
        "--train_data", $TrainData
        "--backbone", $Backbone
        "--learning_rate", $LearningRate
        "--temperature", $Temperature
        "--class_weights", $classWeights
        "--output_dir", $output_dir
    )
    
    # Show Current Experiment Info
    $progress = "[{0}/{1}] Weights: {2}" -f $ExperimentIndex, $TotalExperiments, $configName
    Write-Host $progress -ForegroundColor Cyan
    Write-Host "  Value: $classWeights" -ForegroundColor Gray
    Write-Host "  Note: $($config.Note)" -ForegroundColor Gray
    Write-Host "  Output: $output_dir" -ForegroundColor Gray
    
    # Log Execution
    Log-Message "Starting: $configName"
    Log-Message "  Weights: $classWeights"
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
        Log-Message "  [OK] Experiment Finished: $configName"
        Write-Host "  [OK] Finished" -ForegroundColor Green
    }
    else {
        $FailedExperiments++
        Log-Message "  [FAIL] Experiment Failed: $configName (Exit Code: $exitCode)"
        Write-Host "  [FAIL] Failed (Exit Code: $exitCode)" -ForegroundColor Red
    }
    
    Log-Message ""
    Write-Host ""
}

# Summary
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "========================================" -ForegroundColor Green
Write-Host "Class Weights Search Complete" -ForegroundColor Green
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
Write-Host "Next Step: Compare validation accuracy to find optimal class weights" -ForegroundColor Cyan
Write-Host ""
