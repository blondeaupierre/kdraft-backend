from src.tester import DraftModelEvaluator

evaluator = DraftModelEvaluator(
    model_path="resources/trained_models/NFT_2025-06-01_gpt2_lol_100k",
    tokenizer_path="resources/tokenizer",
    test_dataset_path="resources/datasets/test_top_teams_offline_2025-06-01_dataset.csv",
    max_length=50,
    log_level="INFO"
)
evaluator.save_results()  # Prints results
# evaluator.save_results("results.txt")  # Optionally save to file