python router/train_router.py `
    --config config/train_classification_5000.yaml `
    --train_data "HotpotQA_train_data/label_analysis/all_labels_no_tie_sampled1000.json" `
    --backbone "intfloat/e5-base-v2" `
    --temperature 0.5 `
    --class_weights "no_rag=2.6,naive_rag=1.1" `
    --output_dir "router_models/no_tie_sampled1000_e5"