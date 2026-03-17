import json
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

def load_teams(path: Path):
    """
    Load team names from a .txt or .json file.
    - If .txt → uses load_txt()
    - If .json → loads 'full_name' from each object
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # ensure list of objects
                if isinstance(data, dict):
                    data = [data]
                teams = [item["full_name"] for item in data if "full_name" in item]
                return teams
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file {path}: {e}")
    elif suffix == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def load_partial_sequence(sequence_path: str) -> str:
    try:
        with open(sequence_path, "r", encoding="utf-8") as f:
            line = [line.strip() for line in f if line.strip()]
        flattened = "".join(line)

        return flattened
    except FileNotFoundError:
        raise FileNotFoundError(f"Partial sequence file not found at {sequence_path}")