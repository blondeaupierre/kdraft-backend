import yaml

from src.utils.tokenizer_builder import DraftTokenizerBuilder

with open("config/tokenizer_config.yaml", "r") as f:
    config = yaml.safe_load(f)

builder = DraftTokenizerBuilder(
    vocab_dir=config["vocab_dir"],
    save_dir=config["save_dir"],
    log_level="INFO"
)
builder.save_tokenizer()
