from pathlib import Path

def load_txt(path: Path|str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Draft tokens file not found at {path}")

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
        flattened = "".join(line)

        return flattened
    except FileNotFoundError:
        raise FileNotFoundError(f"Partial sequence file not found at {sequence_path}")