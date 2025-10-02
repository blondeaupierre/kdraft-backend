import logging

import torch
import torch.nn.functional as F
from torch.cuda import device
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, LogitsProcessorList

from src.utils.data_utils import load_partial_sequence, load_txt
from src.utils.logits_processors import StrictForceTagsProcessor, NoDuplicateChampionsProcessor


class DraftModelGenerator:
    """A class to generate a draft sequence."""

    def __init__(
            self,
            model_path: str,
            tokenizer_path: str,
            champions_path: str,
            partial_sequence_path: str,
            draft_tokens_path: str,
            draft_max_length: int,
            log_level: str = "INFO"
    ):
        # Set up logger
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)

        # Validate inputs
        if not all([model_path, tokenizer_path]):
            raise ValueError("model_path, tokenizer_path, and dataset_path must be provided")

        self.logger.info(f"Loading model from {model_path} and tokenizer from {tokenizer_path}")

        # Load model and tokenizer

        self.model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Draft max length
        self.draft_max_length = draft_max_length

        # Partial sequence
        self.partial_sequence = load_partial_sequence(partial_sequence_path)

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")

        self.champions_path = champions_path
        self.draft_tokens_path = draft_tokens_path
        # Logit Processors
        self.tag_processor = StrictForceTagsProcessor(self.tokenizer, self.draft_tokens_path, self.partial_sequence)
        self.no_duplicate_processor = NoDuplicateChampionsProcessor(self.tokenizer, self.champions_path, self.draft_tokens_path, self.partial_sequence)

    def _prepare_inputs(self):
        """Tokenize the partial sequence for generation."""
        inputs = self.tokenizer(
            self.partial_sequence.split(","),
            is_split_into_words=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def generate_sequence(self):
        """Generate the draft sequence applying logits processors."""
        inputs = self._prepare_inputs()
        logits_processor = LogitsProcessorList([self.tag_processor, self.no_duplicate_processor])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.draft_max_length,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                repetition_penalty=2.0,
                logits_processor=logits_processor
            )

        return outputs

    def compute_topk(self, outputs, top_k):
        """Compute top-k champion tokens and probabilities for each position, including selected token."""
        first_seq = outputs.sequences[0]  # [batch_size, seq_len]
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(first_seq)

        # Draft slots et champions
        draft_tokens = load_txt(self.draft_tokens_path)
        champion_tokens = load_txt(self.champions_path)
        champion_ids = [self.tokenizer.convert_tokens_to_ids(c) for c in champion_tokens]

        results = []

        prompt_len = first_seq.shape[0] - len(outputs.scores)  # longueur du prompt

        for i, tok in enumerate(decoded_tokens[prompt_len:], start=0):

            if tok in draft_tokens:  # uniquement les slots draft
                current_token = tok
                continue
            logits_step = outputs.scores[i]  # shape [batch_size, vocab_size]
            probs = F.softmax(logits_step[0], dim=-1)  # batch_size=1

            champ_probs = probs[champion_ids]
            top_k_probs, top_k_ids = torch.topk(champ_probs, top_k)
            top_k_champions_only = [champion_tokens[idx] for idx in top_k_ids]

            results.append({
                "token": current_token,  # le slot draft, ex: "BLUE_BAN1"
                "top_k": [
                    {"token": champ, "prob": float(prob.item())}
                    for champ, prob in zip(top_k_champions_only, top_k_probs)
                ]
            })

        return results