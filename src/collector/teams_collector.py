import csv
import json
from pathlib import Path

from mwrogue.esports_client import EsportsClient

from src.utils.data_utils import load_txt

site = EsportsClient("lol")

vocab_dir = Path("../../resources/vocab")
vocab_dir.mkdir(exist_ok=True)

# -----------------------------
# Gather top X teams
# -----------------------------
# top_X_teams = site.cargo_client.query(
#     tables="TournamentResults=TR",
#     fields="TR.Team",
#     where="TR.Place_Number <= '8'"
#     # "AND TR.Tier = 'Offline'"
#           "AND TR.Tier = 'Offline' OR  TR.Tier = 'Online/Offline'"
#           "AND TR.Date >= '2025-04-01'"
#           "AND TR.Team IS NOT NULL",
#     group_by="TR.Team"
# )
#
# teams = sorted({row["Team"] for row in top_X_teams if row["Team"]})
#
# with (vocab_dir / "teams.txt").open("w", encoding="utf-8") as f:
#     f.write("\n".join(teams))
# print(f"{len(teams)} teams saved to vocab/teams.txt")



top_X_teams = site.cargo_client.query(
    tables="TournamentResults=TR, Teams=T",
    join_on="TR.Team=T.Name",
    fields="TR.Team, T.Region, T.Short, TR.Tier",
    where="TR.Place_Number <= '10' "
          "AND TR.Tier = 'Offline' "
          "AND TR.Date >= '2025-08-01' "
          "AND TR.Team IS NOT NULL "
          "AND T.Name IS NOT NULL "
          "AND T.Region IS NOT NULL",
    group_by="TR.Team"
)
teams_json = [
    {
        "short_name": row["Short"],   # ex: "Gen.G"
        "full_name": row["Team"],    # ex: "Gen.G Esports"
        "region": row["Region"]      # ex: "KR"
    }
    for row in top_X_teams
]

# Sauvegarder dans un fichier JSON
output_file = Path("../../resources/data/top_teams_offline_2025-06-01.json")
with output_file.open("w", encoding="utf-8") as f:
    json.dump(teams_json, f, indent=2, ensure_ascii=False)

print(f"{len(teams_json)} teams saved to {output_file}")

# # Créer la liste de dictionnaires
# teams = sorted({row["Team"] for row in top_X_teams if row["Team"]})
#
# path = vocab_dir / "top_teams_Offline_2025-06-01.txt"
# with (path).open("w", encoding="utf-8") as f:
#     f.write("\n".join(teams))
#
# print(f"{len(teams)} teams saved to {path}")