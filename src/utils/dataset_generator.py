import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.data_utils import load_draft_tokens

# Load drafts
drafts_df = pd.read_csv("../../resources/data/drafts_context_tokens.csv")

# Load draft tokens in order
draft_tokens = load_draft_tokens("../resources/data/draft_tokens.txt")

# Generate sequences
sequences = []

def build_sequence(row, as_team, vs_team, patch, side):
    meta_tokens = [
        "[AS_TEAM]", as_team,
        "[VS_TEAM]", vs_team,
        "[SIDE]", side,
        "[PATCH]", str(patch)
    ]
    draft_seq = []
    for token in draft_tokens:
        col_name = token.strip("[]")  # access CSV column
        draft_seq.extend([token, row[col_name]])  # keep brackets in sequence

    return meta_tokens + ["<BOS>"] + draft_seq + ["<EOS>"]

for _, row in drafts_df.iterrows():
    # Blue side perspective
    seq_blue = build_sequence(row, as_team=row["BLUE_TEAM"], vs_team=row["RED_TEAM"], patch=row["PATCH"], side="BLUE")
    sequences.append(seq_blue)

    # Red side perspective
    seq_red = build_sequence(row, as_team=row["RED_TEAM"], vs_team=row["BLUE_TEAM"], patch=row["PATCH"], side="RED")
    sequences.append(seq_red)

print(f"Generated {len(sequences)} sequences.")

# Split into train/val/test
train_val, test = train_test_split(sequences, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)

# Convert lists of tokens into DataFrames
train_df = pd.DataFrame({"sequence": [",".join(seq) for seq in train]})
val_df = pd.DataFrame({"sequence": [",".join(seq) for seq in val]})
test_df = pd.DataFrame({"sequence": [",".join(seq) for seq in test]})

# Save CSVs
train_df.to_csv("../resources/datasets/train_context_tokens.csv", index=False)
val_df.to_csv("../resources/datasets/val_context_tokens.csv", index=False)
test_df.to_csv("../resources/datasets/test_context_tokens.csv", index=False)

print("Saved train/val/test CSVs.")