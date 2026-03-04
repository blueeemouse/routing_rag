<#
.SYNOPSIS
    Model comparison evaluation script

.DESCRIPTION
    Compare two models on training and validation sets: Loss, accuracy, confusion matrix, etc.

.PARAMETER Model1
    First model path

.PARAMETER Model2
    Second model path

.PARAMETER TrainData
    Training data path

.PARAMETER ValData
    Validation data path

.PARAMETER BatchSize
    Batch size, default 32

.PARAMETER Device
    Device, default cuda

.EXAMPLE
    .\scripts\run_compare_models.ps1 -Model1 "router_models/model_a/final" -Model2 "router_models/model_b/final"
    
.EXAMPLE
    .\scripts\run_compare_models.ps1 -Model1 "router_models/model_a/checkpoint_step_100" -Model2 "router_models/model_b/checkpoint_step_100" -TrainData "HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json"
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="First model path")]
    # [string]$Model1,
    [string]$Model1 = 'D:\Develop\all_RAG\routing_rag\router_models\BAAI-bge-base-en-v1.5\no_tie_sampled_1000_non_balanced\dc\weight_decay_search_phaseB\wd_10.0\checkpoint_step_50',
    
    [Parameter(Mandatory=$true, HelpMessage="Second model path")]
    # [string]$Model2,
    [string]$Model2 = 'D:\Develop\all_RAG\routing_rag\router_models\backbone_search_sampled1000\bge-base-en-v1.5\checkpoint_step_50',
    
    [Parameter(HelpMessage="Training data path")]
    [string]$TrainData = "HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json",
    # [string]$TrainData = "HotpotQA_train_data/label_analysis/all_labels_with_tie_converted.json",
    
    [Parameter(HelpMessage="Validation data path")]
    [string]$ValData = "evaluation_results/router_test_labels.json",
    
    [Parameter(HelpMessage="Batch size")]
    [int]$BatchSize = 32,
    
    [Parameter(HelpMessage="Device")]
    [string]$Device = "cuda"
)

# Switch to project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model Comparison Evaluation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model 1: $Model1" -ForegroundColor Yellow
Write-Host "Model 2: $Model2" -ForegroundColor Yellow
Write-Host "Train Data: $TrainData" -ForegroundColor Yellow
Write-Host "Val Data: $ValData" -ForegroundColor Yellow
Write-Host "Batch Size: $BatchSize" -ForegroundColor Yellow
Write-Host "Device: $Device" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Run Python script
python scripts/compare_models.py `
    --model1 $Model1 `
    --model2 $Model2 `
    --train_data $TrainData `
    --val_data $ValData `
    --batch_size $BatchSize `
    --device $Device
