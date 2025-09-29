import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_utils import load_draft_tokens

# Load drafts
drafts_df = pd.read_csv("../resources/data/drafts.csv")

# Load draft tokens in order
draft_tokens = load_draft_tokens("../resources/data/draft_tokens")

# Generate sequences
sequences = []
for _, row in drafts_df.iterrows():
    seq = ["<BOS>"] + [token for pair in zip(draft_tokens, row) for token in pair] + ["<EOS>"]
    sequences.append(seq)

print(f"Generated {len(sequences)} sequences.")

# Split into train/val/test
train_val, test = train_test_split(sequences, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)

# Convert lists of tokens into DataFrames
train_df = pd.DataFrame({"sequence": [",".join(seq) for seq in train]})
val_df = pd.DataFrame({"sequence": [",".join(seq) for seq in val]})
test_df = pd.DataFrame({"sequence": [",".join(seq) for seq in test]})

# Save CSVs
train_df.to_csv("../resources/datasets/train.csv", index=False)
val_df.to_csv("../resources/datasets/val.csv", index=False)
test_df.to_csv("../resources/datasets/test.csv", index=False)

print("Saved train/val/test CSVs.")