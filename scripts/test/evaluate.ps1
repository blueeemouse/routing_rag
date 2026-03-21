# Evaluate NoRAG and NaiveRAG on training data
# Note: Comments cannot be placed in the middle of line continuation

# python ./tests/unified_evaluate_hotpotqa.py `
#     --models no_rag,naive_rag `
#     --naive_rag_index_path "D:\Develop\all_RAG\routing_rag\naive_rag_index_storage_10000_train_samples" `
#     --num_samples 10000 `
#     --delay 0 `
#     --hotpotqa_file "D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_train_v1.1_10000_samples.jsonl"

# Alternative: with GraphRAG
# 可以考虑加上delay为0的
python ./tests/unified_evaluate_hotpotqa.py `
    --models graph_rag `
    --graphrag_work_dir "D:\Develop\all_RAG\routing_rag\graphrag_index_hotpotqa_train_5000_samples_fast" `
    --num_samples 5000 `
    --skip_graphrag_index `
    --hotpotqa_file "D:\Develop\all_RAG\routing_rag\HotpotQA\hotpot_train_v1.1_5000_samples.jsonl"
