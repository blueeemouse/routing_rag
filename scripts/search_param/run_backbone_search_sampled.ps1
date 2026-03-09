# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Backbone Search Script (Sampled 1000, Sqrt Inverse Frequency Weights)
# Uses filtered data (no tie samples) with stratified sampling

# Fixed Hyperparameters
$Temperature = 0.5
$ClassWeights = "no_rag=2.6,naive_rag=1.1"  # Sqrt inverse frequency weights
$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json"

# Backbone candidates to test
$Backbones = @(
    @{
        Name = "intfloat/e5-base-v2"
        HiddenSize = 768
        Note = "E5 model, strong performance"
    },
    @{
        Name = "BAAI/bge-base-en-v1.5"
        HiddenSize = 768
        Note = "BGE model, strong performance"
    },
    @{
        Name = "sentence-transformers/all-mpnet-base-v2"
        HiddenSize = 768
        Note = "MPNet, larger than MiniLM"
    }
)

# Output Directory
$ExperimentsRoot = "router_models/backbone_search_sampled1000"
$LogFile = "$ExperimentsRoot/backbone_search_log.txt"

# Helper Functions
function Log-Message {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $message" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
    Write-Host $message -ForegroundColor Cyan
}

function Get-BackboneShortName {
    param([string]$backboneName)
    $parts = $backboneName -split "/"
    if ($parts.Count -gt 1) {
        return $parts[-1]
    }
    return $backboneName
}

# Main Execution
Write-Host "========================================" -ForegroundColor Green
Write-Host "Backbone Search Script (Sampled 1000)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Temperature: $Temperature (fixed)" -ForegroundColor Yellow
Write-Host "  Class Weights: $ClassWeights (sqrt inverse frequency)" -ForegroundColor Yellow
Write-Host "  Train Data: $TrainData" -ForegroundColor Yellow
Write-Host "  Backbones to test: $($Backbones.Count)" -ForegroundColor Yellow
Write-Host "  Output Dir: $ExperimentsRoot" -ForegroundColor Yellow
Write-Host ""

Write-Host "Backbones:" -ForegroundColor Yellow
foreach ($backbone in $Backbones) {
    Write-Host "  - $($backbone.Name)" -ForegroundColor Gray
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

Log-Message "Backbone Search Config (Sampled 1000):"
Log-Message "  Temperature: $Temperature (fixed)"
Log-Message "  Class Weights: $ClassWeights (sqrt inverse frequency)"
Log-Message "  Train Data: $TrainData"
Log-Message "  Backbone count: $($Backbones.Count)"
Log-Message "  Output Dir: $ExperimentsRoot"
Log-Message ""

# Init Statistics
$TotalExperiments = $Backbones.Count
$CompletedExperiments = 0
$FailedExperiments = 0
$StartTime = Get-Date

Write-Host "Preparing to run $TotalExperiments experiments..." -ForegroundColor Yellow
Write-Host ""

# Experiment Loop
$ExperimentIndex = 0
foreach ($backbone in $Backbones) {
    $ExperimentIndex++
    $backboneName = $backbone.Name
    $shortName = Get-BackboneShortName $backboneName
    
    # Generate Output Directory Name
    $output_dir = "$ExperimentsRoot/$shortName"
    
    # Build parameters
    $params = @(
        "router/train_router.py"
        "--config", "config/train_classification_5000.yaml"
        "--train_data", $TrainData
        "--backbone", $backboneName
        "--temperature", $Temperature
        "--class_weights", $ClassWeights
        "--output_dir", $output_dir
    )
    
    # Show Current Experiment Info
    $progress = "[{0}/{1}] Backbone: {2}" -f $ExperimentIndex, $TotalExperiments, $backboneName
    Write-Host $progress -ForegroundColor Cyan
    Write-Host "  Note: $($backbone.Note)" -ForegroundColor Gray
    Write-Host "  Output: $output_dir" -ForegroundColor Gray
    
    # Log Execution
    Log-Message "Starting: Backbone=$backboneName"
    Log-Message "  Short Name: $shortName"
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
        Log-Message "  [OK] Experiment Finished: $backboneName"
        Write-Host "  [OK] Finished" -ForegroundColor Green
    }
    else {
        $FailedExperiments++
        Log-Message "  [FAIL] Experiment Failed: $backboneName (Exit Code: $exitCode)"
        Write-Host "  [FAIL] Failed (Exit Code: $exitCode)" -ForegroundColor Red
    }
    
    Log-Message ""
    Write-Host ""
}

# Summary
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "========================================" -ForegroundColor Green
Write-Host "Backbone Search Complete" -ForegroundColor Green
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

# Print results hint
Write-Host "To compare results, check:" -ForegroundColor Cyan
foreach ($backbone in $Backbones) {
    $shortName = Get-BackboneShortName $backbone.Name
    Write-Host "  - $ExperimentsRoot/$shortName/final/" -ForegroundColor Gray
}
Write-Host ""
