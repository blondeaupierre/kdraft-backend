import yaml

from src.utils.tokenizer_builder import DraftTokenizerBuilder

builder = DraftTokenizerBuilder(
    vocab_dir="resources/vocab",
    save_dir="resources/tokenizer",
    log_level="INFO"
)
builder.save_tokenizer()
