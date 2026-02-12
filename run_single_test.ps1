# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Single Experiment Test Script
# For quick validation of training workflow

# Configuration
$Temperature = 0.1
$FixedWeight = "no_rag=1.0,naive_rag=1.0"
$Backbone = "sentence-transformers/all-MiniLM-L6-v2"

# Output directory (use test prefix to avoid overwriting)
$OutputDir = "router_models/test_run"
$LogFile = "$OutputDir/test_run_log.txt"

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Clear log
if (Test-Path $LogFile) {
    Clear-Content $LogFile
}

# Record start time
$StartTime = Get-Date
Write-Host "========================================" -ForegroundColor Green
Write-Host "Single Experiment Test" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Temperature: $Temperature" -ForegroundColor Yellow
Write-Host "  Weight: $FixedWeight" -ForegroundColor Yellow
Write-Host "  Backbone: $Backbone" -ForegroundColor Yellow
Write-Host "  Output: $OutputDir" -ForegroundColor Yellow
Write-Host "  Mode: Overfit single batch + 1 epoch" -ForegroundColor Yellow
Write-Host ""

"[$StartTime] Start test" | Out-File -FilePath $LogFile -Encoding UTF8 -Append

# Build parameters
$params = @(
    "router/train_router.py"
    "--config", "config/train_classification_5000.yaml"
    "--backbone", $Backbone
    "--temperature", $Temperature
    "--class_weights", $FixedWeight
    "--output_dir", $OutputDir
    "--overfit_single_batch"
    "--epochs", "1"
)

Write-Host "Command: python $($params -join ' ')" -ForegroundColor Cyan

# Execute training (fixed exit code capture)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$exitCode = 999

try {
    # Use & call operator to get correct exit code
    $command = "python `"$($params -join '" "')`""
    Write-Host "Executing: $command" -ForegroundColor DarkGray
    
    # Direct call and capture exit code
    Invoke-Expression $command
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq $null) {
        $exitCode = 0
    }
    
    Write-Host "Exit Code: $exitCode" -ForegroundColor Cyan
}
catch {
    $exitCode = -1
    Write-Host " [ERROR] Execution failed: $_" -ForegroundColor Red
    "[ERROR] Execution failed: $_" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
}

# End time
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

# Output results
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Test Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($exitCode -eq 0) {
    Write-Host "[OK] Experiment completed successfully!" -ForegroundColor Green
    "[$EndTime] Test completed successfully, duration: $Duration" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
} else {
    Write-Host "[FAIL] Experiment failed (Exit Code: $exitCode)" -ForegroundColor Red
    "[$EndTime] Test failed, Exit Code: $exitCode, duration: $Duration" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
}

Write-Host "Start Time: $StartTime" -ForegroundColor Yellow
Write-Host "End Time: $EndTime" -ForegroundColor Yellow
Write-Host "Duration: $Duration" -ForegroundColor Yellow
Write-Host "Output Directory: $OutputDir" -ForegroundColor Yellow
Write-Host ""
