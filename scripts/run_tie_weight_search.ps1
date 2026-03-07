# Tie Weight Search Script
# Test different tie sample weights and observe training effects

# Output directory
$EXPERIMENTS_ROOT = "router_models/tie_weight_search"
$LOG_FILE = "$EXPERIMENTS_ROOT/tie_weight_search_log.txt"

# Data configuration
$TRAIN_DATA = "HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json"
$VAL_DATA = "evaluation_results/router_test_labels.json"

# Tie weights to search
$TIE_WEIGHTS = @(0.2, 0.5, 1.0, 1.5, 2.0)

# Set GPU
$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "========================================"
Write-Host "Tie Weight Search Script"
Write-Host "========================================"
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Backbone: bge-base-en-v1.5 (feature_fused, lr=7e-5, wd=10)"
Write-Host "  Temperature: 0.5"
Write-Host "  Epochs: 5"
Write-Host "  Tie weights: $($TIE_WEIGHTS -join ', ')"
Write-Host "  Output dir: $EXPERIMENTS_ROOT"
Write-Host "  GPU: 0"
Write-Host ""

# Create output directory
if (-not (Test-Path $EXPERIMENTS_ROOT)) {
    New-Item -ItemType Directory -Force -Path $EXPERIMENTS_ROOT | Out-Null
}

# Initialize log
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Tie weight search started" | Out-File -FilePath $LOG_FILE -Encoding UTF8
"Configuration:" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Backbone: bge-base-en-v1.5 (feature_fused)" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Temperature: 0.5" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Epochs: 5" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Tie weights: $($TIE_WEIGHTS -join ', ')" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append

# Statistics variables
$TOTAL = $TIE_WEIGHTS.Count
$COMPLETED = 0
$FAILED = 0
$START_TIME = Get-Date

Write-Host "Preparing to run $TOTAL experiments..."
Write-Host ""

# ============================================
# Backbone: BAAI/bge-base-en-v1.5 (feature_fused model)
# ============================================
$BACKBONE_NAME = "bge-base-en-v1.5"
$BACKBONE = "BAAI/bge-base-en-v1.5"
$MODEL_TYPE = "feature_fused"
$TRAINER_TYPE = "feature_fused"
$HIDDEN_SIZE = "768"
$LEARNING_RATE = "7e-5"
$TEMPERATURE = "0.5"
$EPOCHS = "5"
$WEIGHT_DECAY = "10"

Write-Host "----------------------------------------"
Write-Host "Backbone: $BACKBONE_NAME (model_type=$MODEL_TYPE, lr=$LEARNING_RATE, wd=$WEIGHT_DECAY)"
Write-Host "----------------------------------------"
Write-Host ""

foreach ($tie_weight in $TIE_WEIGHTS) {
    # Generate output directory name
    $OUTPUT_DIR = "$EXPERIMENTS_ROOT/$BACKBONE_NAME/tie_weight_$tie_weight"
    
    # Display current experiment info
    $progress = "[$($COMPLETED + $FAILED + 1)/$TOTAL] $BACKBONE_NAME, tie_weight=$tie_weight"
    Write-Host $progress
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] Started: $BACKBONE_NAME, tie_weight=$tie_weight" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
    
    # Execute training
    $command = "python router/train_router.py --model_type $MODEL_TYPE --trainer_type $TRAINER_TYPE --backbone $BACKBONE --train_data $TRAIN_DATA --val_data $VAL_DATA --learning_rate $LEARNING_RATE --temperature $TEMPERATURE --epochs $EPOCHS --weight_decay $WEIGHT_DECAY --tie_weight $tie_weight --output_dir $OUTPUT_DIR"
    Write-Host "Command: $command"
    
    Invoke-Expression $command
    
    # Check result
    if ($LASTEXITCODE -eq 0) {
        $COMPLETED++
        Write-Host "[SUCCESS] Experiment completed: $BACKBONE_NAME, tie_weight=$tie_weight"
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$timestamp] [SUCCESS] Experiment completed: $BACKBONE_NAME, tie_weight=$tie_weight" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
    }
    else {
        $FAILED++
        Write-Host "[FAILED] Experiment failed: $BACKBONE_NAME, tie_weight=$tie_weight (exit code: $LASTEXITCODE)"
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$timestamp] [FAILED] Experiment failed: $BACKBONE_NAME, tie_weight=$tie_weight (exit code: $LASTEXITCODE)" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
    }
    
    "" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
    Write-Host ""
}

# Statistics info
$END_TIME = Get-Date
$DURATION = $END_TIME - $START_TIME
$HOURS = [math]::Floor($DURATION.TotalHours)
$MINUTES = $DURATION.Minutes
$SECONDS = $DURATION.Seconds

Write-Host "========================================"
Write-Host "Tie Weight Search Completed"
Write-Host "========================================"
Write-Host ""
Write-Host "Experiment Statistics:"
Write-Host "  Total: $TOTAL"
Write-Host "  Success: $COMPLETED"
Write-Host "  Failed: $FAILED"
Write-Host "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
Write-Host ""

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Search completed" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Total experiments: $TOTAL" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Success: $COMPLETED" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Failed: $FAILED" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Log file: $LOG_FILE" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append

Write-Host "Results saved in: $EXPERIMENTS_ROOT"
Write-Host "Log file: $LOG_FILE"
Write-Host ""
