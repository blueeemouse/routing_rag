# 内部表征路由器训练脚本
# 使用预提取的 LLM 内部表征训练路由器

# 配置参数
$Config = "config/train_internal_representation.yaml"
$ModelType = "internal_representation"
$TrainerType = "internal_representation"

# 数据路径
$TrainData = "outputs/representations/fp16_qwen2.5-3b-instruct"
$ValData = "outputs/representations/fp16_qwen2.5-3b-instruct_test1000"

# 表征类型
$RepresentationType = "deep_last_token"

# 训练超参数
$BatchSize = 32
$LearningRate = "1e-4"
$Epochs = 10
$EvalSteps = 100

# 输出配置
$OutputDir = "outputs/internal_representation_router"
$Seed = 42

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "内部表征路由器训练" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "配置文件: $Config"
Write-Host "模型类型: $ModelType"
Write-Host "表征类型: $RepresentationType"
Write-Host "训练数据: $TrainData"
Write-Host "验证数据: $ValData"
Write-Host "批量大小: $BatchSize"
Write-Host "学习率: $LearningRate"
Write-Host "训练轮数: $Epochs"
Write-Host "输出目录: $OutputDir"
Write-Host "========================================" -ForegroundColor Cyan

# 执行训练
python router/train_router.py `
    --config $Config `
    --model_type $ModelType `
    --trainer_type $TrainerType `
    --train_data $TrainData `
    --val_data $ValData `
    --batch_size $BatchSize `
    --learning_rate $LearningRate `
    --epochs $Epochs `
    --eval_steps $EvalSteps `
    --output_dir $OutputDir `
    --seed $Seed

Write-Host "`n训练完成！" -ForegroundColor Green
Write-Host "模型保存在: $OutputDir/final" -ForegroundColor Green
