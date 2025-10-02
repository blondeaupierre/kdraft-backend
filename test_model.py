import yaml

from src.tester import DraftModelEvaluator

with open("config/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

evaluator = DraftModelEvaluator(
    model_path=config["model_path"],
    tokenizer_path=config["tokenizer_path"],
    test_dataset_path=config["test_dataset_path"],
    max_length=config["max_length"],
    log_level="INFO"
)
# evaluator.save_results()  # Prints results
evaluator.save_results("results.txt")  # Optionally save to file