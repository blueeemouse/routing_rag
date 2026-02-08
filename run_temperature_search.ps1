# 设置控制台输出编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Temperature Debugging Script
# Aims to test different temperature parameters while keeping others fixed

# Define Parameters
 $Temperatures = @(0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
 $FixedWeight = "no_rag=1.0,naive_rag=1.0"
 $Backbone = "sentence-transformers/all-MiniLM-L6-v2"

# Base Arguments
 $BaseArgs = @(
    # "--model_type", "classification",
    "--config", "config/train_classification_5000.yaml"
    # "--train_data", "data/train_router_labels.jsonl",
    # "--val_data", "evaluation_results/router_test_labels.jsonl",
    # "--overfit_single_batch",
    # "--fast_dev_steps", "10"
)

# Output Directory
 $ExperimentsRoot = "router_models/temperature_search"
 $LogFile = "$ExperimentsRoot/temperature_search_log.txt"

# Helper Functions

function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

# Main Execution

Write-Host "========================================" -ForegroundColor Green
Write-Host "Temperature Debugging Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Debugging Config:" -ForegroundColor Yellow
Write-Host "  Temperatures: $($Temperatures -join ', ')"
Write-Host "  Fixed Weight: $FixedWeight (Fixed)"
Write-Host "  Backbone: $Backbone (Fixed)"
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

Log-Message "Temperature Debugging Config:"
Log-Message " Temperatures: $($Temperatures -join ', ')"
Log-Message " Fixed Weight: $FixedWeight (Fixed)"
Log-Message "  Backbone: $Backbone (Fixed)"
Log-Message "  Output Dir: $ExperimentsRoot"
Log-Message "  Log File: $LogFile"

# Init Statistics
 $TotalExperiments = $Temperatures.Count
 $CompletedExperiments = 0
 $FailedExperiments = 0
 $StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow
Write-Host ""

# Experiment Loop
foreach ($temp in $Temperatures) {
    # Generate Output Directory Name
    $output_dir = "$ExperimentsRoot/temp_$temp"

    # $PythonExe = "C:\Users\lanhz\miniconda3\envs\ant-graphrag-dev\python.exe"

    # Construct Full Command（使用数组方式避免PowerShell解析错误）
    $params = @(
        "router/train_router.py"
        $($BaseArgs -split ' ')
        "--backbone", $Backbone
        "--temperature", $temp
        "--class_weights", $FixedWeight
        "--output_dir", $output_dir
    )
    $command = "python $($params -join ' ')"

    # Show Current Experiment Info
    $progress = "[{0}/{1}] Temperature = {2}" -f ($CompletedExperiments + 1), $TotalExperiments, $temp
    Write-Host $progress -ForegroundColor Cyan

    # Log Execution
    Log-Message "Starting: Temp=$temp"
    Log-Message "Command: $command"

    # Execute Training
    # 【关键修改】获取脚本所在目录，并设置为工作目录
    # $MyInvocation.MyCommand.Path 指向 ps1 文件本身，Split-Path -Parent 就是其所在目录
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

    $exitCode = 0
    try {
        # 【关键修改】使用 Start-Process 执行命令，支持 -WorkingDirectory 参数
        $process = Start-Process -FilePath "python" -ArgumentList $params -WorkingDirectory $ScriptDir -Wait -NoNewWindow
        $exitCode = $process.ExitCode
    }
    catch {
        $exitCode = $_.Exception.HResult
        Log-Message " [ERROR] Execution Failed: $_"
    }

    # Check Results
    if ($exitCode -eq 0) {
        $CompletedExperiments++
        Log-Message " [OK] Experiment Finished: Temp=$temp"
    }
    else {
        $FailedExperiments++
        Log-Message " [FAIL] Experiment Failed: Temp=$temp (Exit Code: $exitCode)"
    }

    Log-Message ""
}

# Summary
 $EndTime = Get-Date
 $Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Temperature Debugging Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
# Log-Host "Summary:"
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