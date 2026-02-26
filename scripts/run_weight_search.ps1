# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Weight Search Script
# Tests different class weights while keeping temperature fixed at best value

# Configuration
$Temperature = 0.5  # Best temperature from previous search
$Backbone = "sentence-transformers/all-MiniLM-L6-v2"

# Weight search space: fixed naive_rag=1.0, vary no_rag
# Range: 0.5-3.0 with finer granularity around 1.0-3.0
$NoRagWeights = @(0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0)

# Output Directory
$ExperimentsRoot = "router_models/weight_search"
$LogFile = "$ExperimentsRoot/weight_search_log.txt"

# Helper Functions
function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

# Main Execution
Write-Host "========================================" -ForegroundColor Green
Write-Host "Weight Search Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Temperature: $Temperature (fixed)" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Yellow
Write-Host "  Weight Search: no_rag varied, naive_rag=1.0 (fixed)" -ForegroundColor Yellow
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

Log-Message "Weight Search Config:"
Log-Message "  Temperature: $Temperature (fixed)"
Log-Message "  Backbone: $Backbone"
Log-Message "  Weight combinations: $($NoRagWeights.Count) experiments"
Log-Message "  Output Dir: $ExperimentsRoot"

# Init Statistics
$TotalExperiments = $NoRagWeights.Count
$CompletedExperiments = 0
$FailedExperiments = 0
$StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow
Write-Host ""

# Experiment Loop
foreach ($noRagWeight in $NoRagWeights) {
    $weightStr = "no_rag=$noRagWeight,naive_rag=1.0"
    
    # Generate Output Directory Name
    $output_dir = "$ExperimentsRoot/weight_no${noRagWeight}_naive1.0"
    
    # Build parameters
    $params = @(
        "router/train_router.py"
        "--config", "config/train_classification_5000.yaml"
        "--backbone", $Backbone
        "--temperature", $Temperature
        "--class_weights", $weightStr
        "--output_dir", $output_dir
    )
    
    # Show Current Experiment Info
    $progress = "[{0}/{1}] no_rag={2}, naive_rag=1.0" -f ($CompletedExperiments + $FailedExperiments + 1), $TotalExperiments, $noRagWeight
    Write-Host $progress -ForegroundColor Cyan
    
    # Log Execution
    Log-Message "Starting: no_rag=$noRagWeight, naive_rag=1.0"
    Log-Message "Command: python $($params -join ' ')"
    
    # Execute Training
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
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
        Log-Message " [ERROR] Execution Failed: $_"
    }
    
    # Check Results
    if ($exitCode -eq 0) {
        $CompletedExperiments++
        Log-Message " [OK] Experiment Finished: no_rag=$noRagWeight"
    }
    else {
        $FailedExperiments++
        Log-Message " [FAIL] Experiment Failed: no_rag=$noRagWeight (Exit Code: $exitCode)"
    }
    
    Log-Message ""
}

# Summary
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Weight Search Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Log-Message "  Total Experiments: $TotalExperiments"
Log-Message "  Success: $CompletedExperiments"
Log-Message "  Failed: $FailedExperiments"
Log-Message "  Start Time: $StartTime"
Log-Message "  End Time: $EndTime"
Log-Message "  Duration: $Duration"
Log-Message ""
Log-Message "Detailed logs saved to: $LogFile"

Write-Host ""
Write-Host "Experiment Stats: Success=$CompletedExperiments/$TotalExperiments" -ForegroundColor $(if ($CompletedExperiments -eq $TotalExperiments) { "Green" } else { "Yellow" })
Write-Host ""
Write-Host "Results saved in: $ExperimentsRoot" -ForegroundColor Yellow
Write-Host ""
