import difflib
import json

from src.utils.data_utils import load_txt


def normalize_team_name(input_name: str, teams_json: list[str]) -> str:
    """
        Normalize a team name by finding the closest match in the vocabulary.

        Args:
            input_name (str): Team name from UI or external source.
            teams_json (list[str]): List of valid team names (from vocab file).
        Returns:
            str: The closest team name found in vocab, or the raw input if no match.
        """
    short_names = [team["short_name"] for team in teams_json]
    full_names = [team["full_name"] for team in teams_json]

    match_short = difflib.get_close_matches(input_name, short_names, n=1, cutoff=0.8)
    if match_short:
        for team in teams_json:
            if team["short_name"] == match_short[0]:
                print("match found :", match_short[0])
                return team["full_name"]

    match_full = difflib.get_close_matches(input_name, full_names, n=1, cutoff=0.6)
    if match_full:
        print("match found :", match_full[0])
        return match_full[0]

    print(f"[WARN] No match found for '{input_name}'")
    return input_name


def build_draft_sequence(as_team: str, vs_team: str, side: str, patch: str, draft_sequence: str, top_teams_json_path: str) -> str:
    """
    Build the draft sequence string to feed into the model.
    """
    with open(top_teams_json_path, "r", encoding="utf-8") as f:
        teams_json = json.load(f)

    as_team = normalize_team_name(as_team, teams_json)
    vs_team = normalize_team_name(vs_team, teams_json)

    sequence = f"[AS_TEAM],{as_team},[VS_TEAM],{vs_team},[SIDE],{side},[PATCH],{patch},{draft_sequence}"

    return sequence

build_draft_sequence("fnatic","CFO","cc","cc","cc","../../resources/data/top_teams_offline_2025-06-01.json")