import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import os


def tokenize_csv(
    csv_path: str,
    tokenizer_path: str,
    output_dir: str,
    max_length: int = 50,
    chunk_size: int = 5000
):
    """
    Tokenizes a CSV file of draft sequences into a Hugging Face Dataset and saves it to disk.

    Args:
        csv_path (str): Path to CSV with column "sequence".
        tokenizer_path (str): Path to the custom tokenizer directory.
        output_dir (str): Where to save the tokenized dataset.
        max_length (int): Max token sequence length (constant for LoL drafts = 50).
        chunk_size (int): Number of rows to process at once (to avoid RAM overflow).
    """
    print(f"🚀 Tokenizing dataset: {csv_path}")
    print(f"➡️ Saving tokenized version to: {output_dir}")

    # Load custom tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Create output dir if needed
    os.makedirs(output_dir, exist_ok=True)

    datasets = []

    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc=f"Tokenizing {os.path.basename(csv_path)}"):
        if "sequence" not in chunk.columns:
            raise ValueError("CSV must contain a 'sequence' column")

        # Split comma-separated tokens
        tokenized = tokenizer(
            [seq.split(',') for seq in chunk["sequence"]],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        ds = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        })
        datasets.append(ds)

    # Concatenate all chunks into one dataset
    full_dataset = concatenate_datasets(datasets)
    full_dataset.save_to_disk(output_dir)

    print(f"✅ Saved tokenized dataset with {len(full_dataset)} samples at {output_dir}")


if __name__ == "__main__":
    # Paths
    tokenizer_path = "../../resources/tokenizer"
    base_data_dir = "../../resources/datasets"

    # Files to tokenize
    datasets = {
        "train": f"{base_data_dir}/train_top_teams_offline_2025-06-01_dataset.csv",
        "val": f"{base_data_dir}/val_top_teams_offline_2025-06-01_dataset.csv",
        "test": f"{base_data_dir}/test_top_teams_offline_2025-06-01_dataset.csv",
    }

    # Tokenize all
    for split_name, csv_path in datasets.items():
        output_dir = f"{base_data_dir}/tokenized/tokenized_{split_name}"
        tokenize_csv(csv_path, tokenizer_path, output_dir, max_length=50, chunk_size=5000)
