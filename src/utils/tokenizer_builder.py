import csv
import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

from src.utils.data_utils import load_txt, load_draft_tokens


class DraftTokenizerBuilder:
    """A class to build and save a tokenizer for draft sequences based on GPT-2 small."""

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
        # Charger le tokenizer GPT-2 small
        base_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
        self.logger.info(f"Base tokenizer contains {len(base_tokenizer)} tokens")
        # Construire le tokenizer personnalisé
        all_tokens = self.special_tokens + self.meta_tokens + self.draft_tokens + \
                     self.champions + self.teams + self.patches + self.sides

        # Identifier les tokens qui ne sont pas déjà dans GPT-2
        new_tokens = [tok for tok in all_tokens if tok not in base_tokenizer.get_vocab()]
        self.logger.info(f"Ajout de {len(new_tokens)} nouveaux tokens au vocabulaire GPT-2.")

        # Ajouter les nouveaux tokens au tokenizer GPT-2
        base_tokenizer.add_tokens(new_tokens)

        # Définir les tokens spéciaux
        base_tokenizer.bos_token = "<BOS>"
        base_tokenizer.eos_token = "<EOS>"
        base_tokenizer.pad_token = "<PAD>"
        base_tokenizer.unk_token = "<UNK>"

        self.logger.info(f"Tokenizer final vocab size: {len(base_tokenizer)}")
        return base_tokenizer

    def save_tokenizer(self):
        """Build and save the tokenizer to save_dir."""
        tokenizer = self.build_tokenizer()
        self.logger.info(f"Saving tokenizer to {self.save_dir}")
        tokenizer.save_pretrained(self.save_dir)
