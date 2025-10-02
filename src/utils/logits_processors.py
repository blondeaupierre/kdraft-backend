import torch
from transformers import LogitsProcessor

from src.utils.data_utils import load_txt


def tokenize_prompt(tokenizer, prompt_text):
    """Tokenize prompt_text, handling comma-separated or single-token inputs."""
    prompt_tokens = prompt_text.split(',') if ',' in prompt_text else [prompt_text]
    return tokenizer(prompt_tokens, is_split_into_words=True, return_tensors="pt")["input_ids"]


class StrictForceTagsProcessor(LogitsProcessor):
    """
    Args:
        tokenizer: Hugging Face tokenizer to convert tokens to IDs
        tag_positions: List of positions where tags must appear (e.g., [0, 2, 4, ...])
        tag_tokens: List of draft tags (e.g., ["[BLUE_BAN1]", "[RED]", ...])
        prompt_text: Initial input sequence as a string, must include <BOS>

    Logic:
    - Tokenize prompt_text to compute input_ids and count tags in the prompt.
    - current_len: Number of tokens generated after the prompt (sequence length minus prompt length).
    - prompt_tags_used: Number of tag tokens in the prompt.
    - effective_len: Accounts for tags in the prompt (effective_len = current_len + prompt_tags_used * 2), assuming each tag is followed by a non-tag token.
    - At tag positions, force the corresponding tag by setting its logit to 0.0 and others to -inf.
    """

    def __init__(self, tokenizer, draft_tokens_path, prompt_text):
        if not draft_tokens_path:
            raise ValueError("tag_tokens_ordered cannot be empty")
        if "<BOS>" not in prompt_text:
            raise ValueError("prompt_text must contain <BOS> to locate the draft start")

        self.tokenizer = tokenizer

        self.tag_tokens = load_txt(draft_tokens_path)
        self.tag_ids = [tokenizer.convert_tokens_to_ids(t) for t in self.tag_tokens]

        self.prompt_ids = tokenize_prompt(tokenizer, prompt_text)
        prompt_list = self.prompt_ids[0].tolist()

        # Get <BOS> tag
        bos_pos = prompt_list.index(tokenizer.bos_token_id)

        # From <BOS> tag, mark the index of the [XXX] tokens, with their appropriate ID
        self.absolute_tag_positions = {
            bos_pos + 1 + 2 * i: tag_id for i, tag_id in enumerate(self.tag_ids)
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        input_ids: (batch, cur_len). scores: (batch, vocab_size) for the next token
        """
        next_pos = input_ids.shape[1]
        if next_pos in self.absolute_tag_positions:
            tag_id = self.absolute_tag_positions[next_pos]
            new_scores = torch.full_like(scores, -1e9)
            new_scores[:, tag_id] = 0.0
            return new_scores
        return scores


"""
Args:
    tokenizer: Hugging Face tokenizer to convert tokens to IDs
    tag_positions: List of positions where tags must appear (to skip champion suppression)
    champion_tokens: List of champion names (e.g., ["Ahri", "Lux", ...])
    prompt_text: Initial input sequence as a string, must include <BOS>

Logic:
- Tokenize prompt_text to compute input_ids and count champions in the prompt.
- current_len: Number of tokens generated after the prompt.
- prompt_champions_used: Number of champion tokens in the prompt.
- effective_len: Accounts for champions in the prompt (effective_len = current_len + prompt_champions_used * 2).
- At non-tag positions, suppress logits of used champions for each sequence by setting them to -inf.
"""


class NoDuplicateChampionsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, champion_path, draft_tokens_path, prompt_text):
        if "<BOS>" not in prompt_text:
            raise ValueError("prompt_text must contain <BOS>")

        self.tokenizer = tokenizer

        self.tag_tokens = load_txt(draft_tokens_path)

        self.champion_tokens = load_txt(champion_path)
        self.champion_ids = [tokenizer.convert_tokens_to_ids(champ) for champ in self.champion_tokens]

        self.prompt_ids = tokenize_prompt(tokenizer, prompt_text)
        prompt_champions_used = self.prompt_ids[0].tolist()

        bos_pos = prompt_champions_used.index(tokenizer.bos_token_id)

        # From <BOS> tag, mark the index of the [XXX] tokens, with their appropriate ID
        self.absolute_tag_positions_set = {
            bos_pos + 1 + 2 * i for i in range(len(self.tag_tokens))
        }

    def __call__(self, input_ids, scores):
        next_pos = input_ids.shape[1]

        # If on a [XXX] token, skip
        if next_pos in self.absolute_tag_positions_set:
            return scores

        new_scores = scores.clone()
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            seq_ids = set(input_ids[i].tolist())
            used_champs = [cid for cid in self.champion_ids if cid in seq_ids]
            if used_champs:
                new_scores[i, used_champs] = -1e9

        return new_scores
