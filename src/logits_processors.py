from transformers import LogitsProcessor
import torch

def tokenize_prompt(tokenizer, prompt_text):
    """Tokenize prompt_text, handling comma-separated or single-token inputs."""
    prompt_tokens = prompt_text.split(',') if ',' in prompt_text else [prompt_text]
    return tokenizer(prompt_tokens, is_split_into_words=True, return_tensors="pt")["input_ids"]

"""
Args:
    tokenizer: Hugging Face tokenizer to convert tokens to IDs
    tag_positions: List of positions where tags must appear (e.g., [0, 2, 4, ...])
    tag_tokens: List of draft tags (e.g., ["<TEAM1_BAN1>", "<TEAM2_BAN1>", ...])
    prompt_text: Initial input sequence as a string, must include <BOS>

Logic:
- Tokenize prompt_text to compute input_ids and count tags in the prompt.
- current_len: Number of tokens generated after the prompt (sequence length minus prompt length).
- prompt_tags_used: Number of tag tokens in the prompt.
- effective_len: Accounts for tags in the prompt (effective_len = current_len + prompt_tags_used * 2), assuming each tag is followed by a non-tag token.
- At tag positions, force the corresponding tag by setting its logit to 0.0 and others to -inf.
"""
class StrictForceTagsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, tag_positions, tag_tokens, prompt_text):
        if not tag_tokens:
            raise ValueError("tag_tokens cannot be empty")
        if not prompt_text.strip():
            raise ValueError("prompt_text cannot be empty")
        if "<BOS>" not in prompt_text:
            raise ValueError("prompt_text must contain <BOS>")

        self.tokenizer = tokenizer
        self.tag_positions = tag_positions
        self.tag_ids = [tokenizer.convert_tokens_to_ids(tag) for tag in tag_tokens]

        if any(tid is None for tid in self.tag_ids):
            raise ValueError("Some tag_tokens are not in the tokenizer's vocabulary")

        self.position_to_tag = dict(zip(tag_positions, self.tag_ids))
        self.prompt_ids = tokenize_prompt(tokenizer, prompt_text)
        self.prompt_tags_used = sum(1 for tid in self.prompt_ids[0] if tid in self.tag_ids)

    def __call__(self, input_ids, scores):
        # Calculate tokens generated after the prompt
        current_len = input_ids.shape[1] - self.prompt_ids.shape[1]
        # Adjust for tags in the prompt (each tag advances position by 2)
        effective_len = current_len + self.prompt_tags_used * 2

        # Force tag at specified positions
        if effective_len in self.tag_positions:
            scores = scores.clone()  # Avoid modifying the original tensor
            scores[:, :] = -float('inf')
            scores[:, self.position_to_tag[effective_len]] = 0.0
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
    def __init__(self, tokenizer, tag_positions, champion_tokens, prompt_text):
        if not champion_tokens:
            raise ValueError("champion_tokens cannot be empty")
        if not prompt_text.strip():
            raise ValueError("prompt_text cannot be empty")
        if "<BOS>" not in prompt_text:
            raise ValueError("prompt_text must contain <BOS>")

        self.tokenizer = tokenizer
        self.tag_positions = tag_positions
        self.champion_ids = [tokenizer.convert_tokens_to_ids(champ) for champ in champion_tokens]

        if any(cid is None for cid in self.champion_ids):
            raise ValueError("Some champion_tokens are not in the tokenizer's vocabulary")

        self.prompt_ids = tokenize_prompt(tokenizer, prompt_text)
        self.prompt_champions_used = sum(1 for tid in self.prompt_ids[0] if tid in self.champion_ids)

    def __call__(self, input_ids, scores):
        # Calculate tokens generated after the prompt
        current_len = input_ids.shape[1] - self.prompt_ids.shape[1]
        # Adjust for champions in the prompt (each champion advances position by 2)
        effective_len = current_len + self.prompt_champions_used * 2

        # Suppress used champions at non-tag positions
        if effective_len not in self.tag_positions:
            scores = scores.clone()  # Avoid modifying the original tensor
            for i in range(input_ids.shape[0]):
                used_champion_ids = [tid.item() for tid in input_ids[i] if tid in self.champion_ids]
                for champ_id in used_champion_ids:
                    scores[i, champ_id] = -float('inf')
        return scores