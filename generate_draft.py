import yaml

from src.generator import DraftModelGenerator

with open("config/generator_config.yaml", "r") as f:
    config = yaml.safe_load(f)

generator = DraftModelGenerator(
    model_path=config["model_path"],
    tokenizer_path=config["tokenizer_path"],
    champions_path=config["champions_path"],
    partial_sequence_path=config["partial_sequence_path"],
    draft_tokens_path=config["draft_tokens_path"],
    draft_max_length=config["draft_max_length"]
)
outputs = generator.generate_sequence()
result_topk = generator.compute_topk(outputs, config["top_k"])
print(result_topk)
