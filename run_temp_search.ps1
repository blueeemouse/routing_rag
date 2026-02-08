# 设置控制台输出编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Temperature Debugging Script

# Parameters
 $Temperatures = @(0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
 $FixedWeight = "no_rag=1.0,naive_rag=1.0"
 $Backbone = "sentence-transformers/all-MiniLM-L6-v2"

# Base Args
 $BaseArgs = @(
    "--model_type", "classification",
    "--config", "config/train_classification_5000.yaml",
    "--train_data", "data/train_router_labels.jsonl",
    "--val_data", "evaluation_results/router_test_labels.jsonl",
    "--overfit_single_batch",
    "--fast_dev_steps", "10"
)

# Output Directory
 $ExperimentsRoot = "router_models/temperature_search"
 $LogFile = "$ExperimentsRoot/temperature_search_log.txt"

# Functions
function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Append -Encoding UTF8
    Write-Host $message
}

# Main
Write-Host "========================================" -ForegroundColor Green
Write-Host "Temperature Debugging Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Setup Directory
if (-not (Test-Path $ExperimentsRoot)) {
    New-Item -ItemType Directory -Path $ExperimentsRoot -Force | Out-Null
}

# Clear Log
if (Test-Path $LogFile) {
    Clear-Content $LogFile
}

# Init Stats
 $TotalExperiments = $Temperatures.Count
 $CompletedExperiments = 0
 $FailedExperiments = 0
 $StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow

# Loop
foreach ($temp in $Temperatures) {
    $output_dir = "$ExperimentsRoot/temp_$temp"
    $command = "python train_router.py $($BaseArgs -join ' ') --backbone $Backbone --temperature $temp --class_weights ""$FixedWeight"" --output_dir ""$output_dir"""
    
    # Progress
    $progress = "Temp=$temp [$CompletedExperiments/$TotalExperiments]"
    Write-Host $progress -ForegroundColor Cyan
    
    # Log
    Log-Message "Starting: Temp=$temp"
    Log-Message "Command: $command"
    
    # Run
    $exitCode = 0
    try {
        & python $command.Split(' ')
        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = $_.Exception.HResult
        Log-Message "ERROR: Execution Failed: $_"
    }

    # Check Result
    if ($exitCode -eq 0) {
        $CompletedExperiments++
        Log-Message "OK: Temp=$temp"
    }
    else {
        $FailedExperiments++
        Log-Message "FAIL: Temp=$temp (Exit Code: $exitCode)"
    }
}

# Summary
 $EndTime = Get-Date
 $Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Temperature Debugging Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Summary:"
Write-Host "  Total Experiments: $TotalExperiments"
Write-Host "  Success: $CompletedExperiments"
Write-Host "  Failed: $FailedExperiments"
Write-Host "  Start Time: $StartTime"
Write-Host "  End Time: $EndTime"
Write-Host "  Duration: $Duration"
Write-Host ""
Write-Host "Detailed logs in: $LogFile"
Write-Host ""
Write-Host "Experiment Stats: Success=$CompletedExperiments/$TotalExperiments" -ForegroundColor $(if ($CompletedExperiments -eq $TotalExperiments) { "Green" } else { "Yellow" })