import logging

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from src.data_utils import load_champions, load_draft_tokens


class DraftTokenizerBuilder:
    """A class to build and save a WordLevel tokenizer for draft sequences."""

    def __init__(
            self,
            champions_path: str,
            draft_tokens_path: str,
            save_dir: str,
            log_level: str = "INFO"
    ):
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)

        if not all([champions_path, draft_tokens_path, save_dir]):
            raise ValueError("champions_path, draft_tokens_path and save_dir paths must be provided")

        # Load champions and draft_tokens
        self.champions = load_champions(champions_path)
        self.draft_tokens = load_draft_tokens(draft_tokens_path)
        self.save_dir = save_dir

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        # Gather all tokens
        special_tokens = ["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
        all_tokens = special_tokens + self.champions + self.draft_tokens
        self.logger.debug(f"Vocabulary: {all_tokens}")

        # Map tokens → ids
        vocab = {tok: i for i, tok in enumerate(all_tokens)}
        self.logger.debug(f"Vocabulary: {vocab}")

        # Build a WordLevel tokenizer
        base_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<UNK>"))

        # Wrap it for Hugging Face
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            unk_token="<UNK>",
            pad_token="<PAD>",
            bos_token="<BOS>",
            eos_token="<EOS>",
        )

        self.logger.info(f"Tokenizer created with vocab size: {tokenizer.vocab_size}")
        return tokenizer

    def save_tokenizer(self):
        """Build and save the tokenizer to save_dir."""
        tokenizer = self.build_tokenizer()
        self.logger.info(f"Saving tokenizer to {self.save_dir}")
        tokenizer.save_pretrained(self.save_dir)
