# python .\tests\unified_evaluate_hotpotqa.py --models \
# --graphrag_work_dir "D:\Develop\all_RAG\routing_rag\graphrag_ollama_hotpotqa_1000_test_data" \
# --num_samples 1000 \
# --skip_graphrag_index

python ./tests/unified_evaluate_hotpotqa.py \
    --models graph_rag \
    --graphrag_work_dir "/home/lhz/code/routing_rag/graphrag_index_hotpotqa_train_5000_samples_fast" \
    --num_samples 5000 \
    --skip_graphrag_index \
    --delay 0 \
    --hotpotqa_file "/home/lhz/code/routing_rag/HotpotQA/hotpot_train_v1.1_5000_samples.jsonl"
