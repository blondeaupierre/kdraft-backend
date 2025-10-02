import csv
import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from src.utils.data_utils import load_txt, load_draft_tokens


class DraftTokenizerBuilder:
    """A class to build and save a WordLevel tokenizer for draft sequences."""

    def __init__(
            self,
            vocab_dir: str,
            save_dir: str,
            log_level: str = "INFO"
    ):
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)

        vocab_dir = Path(vocab_dir)
        self.save_dir = save_dir

        if not vocab_dir.exists():
            raise FileNotFoundError(f"Vocab directory not found: {vocab_dir}")

        self.champions = load_txt(vocab_dir / "champions.txt")
        self.draft_tokens = load_txt(vocab_dir / "draft_tokens.txt")
        self.teams = load_txt(vocab_dir / "teams.txt")
        self.patches = load_txt(vocab_dir / "patches.txt")
        self.special_tokens = load_txt(vocab_dir / "special_tokens.txt")
        self.meta_tokens = load_txt(vocab_dir / "meta_tokens.txt")
        self.sides = ["BLUE", "RED"]

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        # Gather all tokens
        all_tokens = self.special_tokens + self.meta_tokens + self.draft_tokens + self.champions + self.teams + self.patches + self.sides
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
