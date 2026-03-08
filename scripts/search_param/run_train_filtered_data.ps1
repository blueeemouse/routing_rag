# Train Router with Filtered Data (No-RAG strictly better samples removed)
# Dataset: filtered_no_tie_no_rag_strictly_better.json
# - Kept: naive_rag strictly better samples (2090)
# - Kept: tie samples marked as no_rag (2549)
# - Removed: no_rag strictly better samples (361)

# Output directory
$EXPERIMENTS_ROOT = "router_models/filtered_data_training"
$LOG_FILE = "$EXPERIMENTS_ROOT/training_log.txt"

# Data configuration
$TRAIN_DATA = "HotpotQA_train_data/label_analysis/all_labels_no_rag_strictly_better_removed.json"
$VAL_DATA = "evaluation_results/router_test_labels.json"

# Training configuration
$EPOCHS = 5
$TIE_WEIGHT = 1.0

# Set GPU
$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "========================================"
Write-Host "Router Training with Filtered Data"
Write-Host "========================================"
Write-Host ""
Write-Host "Dataset Info:"
Write-Host "  Total samples: ~4639 (2090 naive_rag + 2549 tie)"
Write-Host "  Removed: 361 samples (no_rag strictly better)"
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Backbone 1: all-MiniLM-L6-v2 (dc, config=train_classification_5000.yaml)"
Write-Host "  Backbone 2: bge-base-en-v1.5 (feature_fused, config=train_feature_fused.yaml)"
Write-Host "  Epochs: $EPOCHS"
Write-Host "  Tie weight: $TIE_WEIGHT"
Write-Host "  Output dir: $EXPERIMENTS_ROOT"
Write-Host "  GPU: 0"
Write-Host ""

# Create output directory
if (-not (Test-Path $EXPERIMENTS_ROOT)) {
    New-Item -ItemType Directory -Force -Path $EXPERIMENTS_ROOT | Out-Null
}

# Initialize log
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Training started with filtered data" | Out-File -FilePath $LOG_FILE -Encoding UTF8
"Dataset: all_labels_no_rag_strictly_better_removed.json" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Configuration:" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Backbone 1: all-MiniLM-L6-v2 (dc)" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Backbone 2: bge-base-en-v1.5 (feature_fused)" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Epochs: $EPOCHS" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"  Tie weight: $TIE_WEIGHT" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append

# Statistics
$TOTAL = 2  # 2 backbones
$COMPLETED = 0
$FAILED = 0
$START_TIME = Get-Date

# ============================================
# Backbone 1: all-MiniLM-L6-v2 (dc model)
# ============================================
$BACKBONE_NAME = "all-MiniLM-L6-v2"
$CONFIG_FILE = "config/train_classification_5000.yaml"
$OUTPUT_DIR = "$EXPERIMENTS_ROOT/$BACKBONE_NAME"

Write-Host "----------------------------------------"
Write-Host "[1/$TOTAL] Training: $BACKBONE_NAME (dc)"
Write-Host "----------------------------------------"
Write-Host ""

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Started: $BACKBONE_NAME" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append

$command = "python router/train_router.py --config $CONFIG_FILE --train_data $TRAIN_DATA --val_data $VAL_DATA --epochs $EPOCHS --tie_weight $TIE_WEIGHT --output_dir $OUTPUT_DIR"
Write-Host "Command: $command"

Invoke-Expression $command

if ($LASTEXITCODE -eq 0) {
    $COMPLETED++
    Write-Host "[SUCCESS] Training completed: $BACKBONE_NAME"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] [SUCCESS] Training completed: $BACKBONE_NAME" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
}
else {
    $FAILED++
    Write-Host "[FAILED] Training failed: $BACKBONE_NAME (exit code: $LASTEXITCODE)"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] [FAILED] Training failed: $BACKBONE_NAME (exit code: $LASTEXITCODE)" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
}

"" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
Write-Host ""

# ============================================
# Backbone 2: BAAI/bge-base-en-v1.5 (feature_fused model)
# ============================================
$BACKBONE_NAME = "bge-base-en-v1.5"
$CONFIG_FILE = "config/train_feature_fused.yaml"
$OUTPUT_DIR = "$EXPERIMENTS_ROOT/$BACKBONE_NAME"

Write-Host "----------------------------------------"
Write-Host "[2/$TOTAL] Training: $BACKBONE_NAME (feature_fused)"
Write-Host "----------------------------------------"
Write-Host ""

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Started: $BACKBONE_NAME" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append

$command = "python router/train_router.py --config $CONFIG_FILE --train_data $TRAIN_DATA --val_data $VAL_DATA --epochs $EPOCHS --tie_weight $TIE_WEIGHT --output_dir $OUTPUT_DIR"
Write-Host "Command: $command"

Invoke-Expression $command

if ($LASTEXITCODE -eq 0) {
    $COMPLETED++
    Write-Host "[SUCCESS] Training completed: $BACKBONE_NAME"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] [SUCCESS] Training completed: $BACKBONE_NAME" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
}
else {
    $FAILED++
    Write-Host "[FAILED] Training failed: $BACKBONE_NAME (exit code: $LASTEXITCODE)"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] [FAILED] Training failed: $BACKBONE_NAME (exit code: $LASTEXITCODE)" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
}

"" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append

# Summary
$END_TIME = Get-Date
$DURATION = $END_TIME - $START_TIME
$HOURS = [math]::Floor($DURATION.TotalHours)
$MINUTES = $DURATION.Minutes
$SECONDS = $DURATION.Seconds

Write-Host "========================================"
Write-Host "Training Completed"
Write-Host "========================================"
Write-Host ""
Write-Host "Statistics:"
Write-Host "  Total: $TOTAL"
Write-Host "  Success: $COMPLETED"
Write-Host "  Failed: $FAILED"
Write-Host "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
Write-Host ""

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Training completed" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Total: $TOTAL" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Success: $COMPLETED" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Failed: $FAILED" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append
"" | Out-File -FilePath $LOG_FILE -Encoding UTF8 -Append

Write-Host "Results saved in: $EXPERIMENTS_ROOT"
Write-Host "Log file: $LOG_FILE"
Write-Host ""
