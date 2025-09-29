import yaml

from src.draft_tokenizer_builder import DraftTokenizerBuilder

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

builder = DraftTokenizerBuilder(
    champions_path=config["champions_path"],
    draft_tokens_path=config["draft_tokens_path"],
    save_dir=config["tokenizer_path"],
    log_level="INFO"
)
builder.save_tokenizer()
