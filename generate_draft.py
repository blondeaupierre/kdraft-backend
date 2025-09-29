import pandas as pd
import torch
import yaml
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from torch.nn import functional as F
from transformers import LogitsProcessorList

from src.data_utils import load_draft_tokens, load_champions, load_partial_sequence
from src.logits_processors import StrictForceTagsProcessor, NoDuplicateChampionsProcessor

with open("config/config_finetune.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Load model and tokenizer ---
model = GPT2LMHeadModel.from_pretrained(config["model_path"], local_files_only=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained(config["tokenizer_path"])
tag_tokens = load_draft_tokens(config["draft_tokens_path"])
champions = load_champions(config["champions_path"])
draft_max_length = config["max_length"]

# Define tag positions
tag_positions = list(range(0, len(tag_tokens) * 2, 2))

# --- Setup input ---
partial_sequence = load_partial_sequence(config["partial_sequence_path"])

inputs = tokenizer(
    partial_sequence.split(","),
    is_split_into_words=True,
    return_tensors="pt",
)

device = "cuda" if model.device.type == "cuda" else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

tag_processor = StrictForceTagsProcessor(tokenizer, tag_positions, tag_tokens, partial_sequence)
no_duplicate_processor = NoDuplicateChampionsProcessor(tokenizer, tag_positions, champions, partial_sequence)
logits_processor = LogitsProcessorList([tag_processor, no_duplicate_processor])

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=draft_max_length,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        repetition_penalty=2.0,
        logits_processor=logits_processor,
    )

sequences = outputs.sequences

for i, seq in enumerate(sequences):
    decoded = tokenizer.decode(seq, skip_special_tokens=False)
    print(f"\nGenerated #{i + 1}: {decoded}")

# --- Probability analysis ---
print("\nTop-3 tokens and probabilities for each position (first sequence):")

# Ici tu peux directement utiliser outputs.scores (logits pas à pas)
first_seq = sequences[0]
decoded_tokens = tokenizer.convert_ids_to_tokens(first_seq)

for pos, logits_step in enumerate(outputs.scores):
    # logits_step: [batch_size, vocab_size]
    probs = F.softmax(logits_step[0], dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k=3)
    top_k_tokens = [tokenizer.decode([id_]) for id_ in top_k_ids]

    print(f"\nPosition {pos} (after token '{decoded_tokens[pos]}'):")
    for tok, prob in zip(top_k_tokens, top_k_probs):
        print(f"  Token: {tok}, Probability: {prob.item():.4f}")