# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Learning Rate Search - Phase 1B: Fine-grained Search
# Based on Phase 1A results: 1e-4 (0.551) > 1e-5 (0.533) > 1e-6 (0.45) > 1e-3 (0.43)
# Focus on 1e-5 ~ 3e-4 range

# Fixed Hyperparameters
$Backbone = "BAAI/bge-base-en-v1.5"
$Temperature = 0.5
$ClassWeights = "no_rag=2.6,naive_rag=1.1"  # Sqrt inverse frequency weights
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json"

# Learning rates to test (fine-grained, focused on best range from Phase 1A)
$LearningRates = @("1e-5", "3e-5", "5e-5", "7e-5", "9e-5", "1e-4", "2e-4", "3e-4")

# Output Directory
$ExperimentsRoot = "router_models/lr_search_phase1b"
$LogFile = "$ExperimentsRoot/lr_search_log.txt"

# Helper Functions
function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

# Main Execution
Write-Host "========================================" -ForegroundColor Green
Write-Host "Learning Rate Search - Phase 1B (Fine-grained)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Phase 1A Results:" -ForegroundColor Yellow
Write-Host "  1e-4: 0.551 (best)" -ForegroundColor Gray
Write-Host "  1e-5: 0.533" -ForegroundColor Gray
Write-Host "  1e-6: 0.45" -ForegroundColor Gray
Write-Host "  1e-3: 0.43" -ForegroundColor Gray
Write-Host ""
Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Yellow
Write-Host "  Temperature: $Temperature (fixed)" -ForegroundColor Yellow
Write-Host "  Class Weights: $ClassWeights" -ForegroundColor Yellow
Write-Host "  Train Data: $TrainData" -ForegroundColor Yellow
Write-Host "  Learning Rates: $($LearningRates -join ', ')" -ForegroundColor Yellow
Write-Host "  Output Dir: $ExperimentsRoot" -ForegroundColor Yellow
Write-Host ""

# Create Output Directory
if (-not (Test-Path $ExperimentsRoot)) {
    New-Item -ItemType Directory -Path $ExperimentsRoot -Force | Out-Null
}

# Clear Log File
if (Test-Path $LogFile) {
    Clear-Content $LogFile
}

Log-Message "Learning Rate Search Phase 1B Config:"
Log-Message "  Backbone: $Backbone"
Log-Message "  Temperature: $Temperature"
Log-Message "  Class Weights: $ClassWeights"
Log-Message "  Train Data: $TrainData"
Log-Message "  Learning Rates: $($LearningRates -join ', ')"
Log-Message "  Output Dir: $ExperimentsRoot"
Log-Message ""

# Init Statistics
$TotalExperiments = $LearningRates.Count
$CompletedExperiments = 0
$FailedExperiments = 0
$StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow
Write-Host ""

# Experiment Loop
$ExperimentIndex = 0
foreach ($lr in $LearningRates) {
    $ExperimentIndex++
    
    # Generate Output Directory Name
    $lrSafe = $lr -replace '-', 'n'  # 1e-5 -> 1en5
    $output_dir = "$ExperimentsRoot/lr_$lrSafe"
    
    # Build parameters
    $params = @(
        "router/train_router.py"
        "--config", "config/train_classification_5000.yaml"
        "--train_data", $TrainData
        "--backbone", $Backbone
        "--learning_rate", $lr
        "--temperature", $Temperature
        "--class_weights", $ClassWeights
        "--output_dir", $output_dir
    )
    
    # Show Current Experiment Info
    $progress = "[{0}/{1}] Learning Rate: {2}" -f $ExperimentIndex, $TotalExperiments, $lr
    Write-Host $progress -ForegroundColor Cyan
    Write-Host "  Output: $output_dir" -ForegroundColor Gray
    
    # Log Execution
    Log-Message "Starting: LR=$lr"
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
        Log-Message "  [OK] Experiment Finished: LR=$lr"
        Write-Host "  [OK] Finished" -ForegroundColor Green
    }
    else {
        $FailedExperiments++
        Log-Message "  [FAIL] Experiment Failed: LR=$lr (Exit Code: $exitCode)"
        Write-Host "  [FAIL] Failed (Exit Code: $exitCode)" -ForegroundColor Red
    }
    
    Log-Message ""
    Write-Host ""
}

# Summary
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "========================================" -ForegroundColor Green
Write-Host "Learning Rate Search Phase 1B Complete" -ForegroundColor Green
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
Write-Host "Next Step: Compare validation accuracy to find optimal learning rate" -ForegroundColor Cyan
Write-Host ""
