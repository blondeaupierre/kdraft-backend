import pandas as pd

def load_champions(champions_path: str) -> list:
    try:
        df = pd.read_csv(champions_path)
        return df.iloc[:, 0].tolist()
    except FileNotFoundError:
        raise FileNotFoundError(f"Champions file not found at {champions_path}")

def load_draft_tokens(tokens_path: str) -> list:
    try:
        with open(tokens_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Draft tokens file not found at {tokens_path}")

def load_partial_sequence(sequence_path: str) -> str:
    try:
        with open(sequence_path, "r", encoding="utf-8") as f:
            line = [line.strip() for line in f if line.strip()]

        if not line:
            return "<BOS>"

        flattened = "".join(line)
        return flattened
    except FileNotFoundError:
        raise FileNotFoundError(f"Partial sequence file not found at {sequence_path}")