import csv
from pathlib import Path

from mwrogue.esports_client import EsportsClient

from src.utils.data_utils import load_txt

site = EsportsClient("lol")

vocab_dir = Path("../../resources/vocab")
vocab_dir.mkdir(exist_ok=True)

output_file = "../../resources/data/drafts_context_tokens.csv"

# -----------------------------
# Get the drafts
# -----------------------------
teams = load_txt(vocab_dir / "teams.txt")
escaped = [t.replace("'", "''") for t in teams]
teams_filter = "('" + "','".join(escaped) + "')"

drafts = site.cargo_client.query(
    tables="MatchSchedule=MS, PicksAndBansS7=PAB",
    join_on="MS.MatchId=PAB.MatchId",
    fields="PAB.Team1, "
           "PAB.Team2, "
           "MS.Patch, "
           "PAB.Team1Ban1, "
           "PAB.Team2Ban1, "
           "PAB.Team1Ban2, "
           "PAB.Team2Ban2, "
           "PAB.Team1Ban3, "
           "PAB.Team2Ban3, "
           "PAB.Team1Pick1, "
           "PAB.Team2Pick1, "
           "PAB.Team2Pick2, "
           "PAB.Team1Pick2, "
           "PAB.Team1Pick3, "
           "PAB.Team2Pick3, "
           "PAB.Team2Ban4, "
           "PAB.Team1Ban4, "
           "PAB.Team2Ban5, "
           "PAB.Team1Ban5, "
           "PAB.Team2Pick4, "
           "PAB.Team1Pick4, "
           "PAB.Team1Pick5, "
           "PAB.Team2Pick5"
    ,
    where="MS.Patch>='12.10' "
          "AND PAB.Team1 IS NOT NULL AND PAB.Team1 <> 'None' AND PAB.Team1 <> 'Missing Data'"
          "AND PAB.Team2 IS NOT NULL AND PAB.Team2 <> 'None' AND PAB.Team2 <> 'Missing Data'"

          "AND PAB.Team1Ban1 IS NOT NULL AND PAB.Team1Ban1 <> 'None' AND PAB.Team1Ban1 <> 'Missing Data'"
          "AND PAB.Team1Ban2 IS NOT NULL AND PAB.Team1Ban2 <> 'None' AND PAB.Team1Ban2 <> 'Missing Data'"
          "AND PAB.Team1Ban3 IS NOT NULL AND PAB.Team1Ban3 <> 'None' AND PAB.Team1Ban3 <> 'Missing Data'"
          "AND PAB.Team1Ban4 IS NOT NULL AND PAB.Team1Ban4 <> 'None' AND PAB.Team1Ban4 <> 'Missing Data'"
          "AND PAB.Team1Ban5 IS NOT NULL AND PAB.Team1Ban5 <> 'None' AND PAB.Team1Ban5 <> 'Missing Data'"
          ""
          "AND PAB.Team2Ban1 IS NOT NULL AND PAB.Team2Ban1 <> 'None' AND PAB.Team2Ban1 <> 'Missing Data' "
          "AND PAB.Team2Ban2 IS NOT NULL AND PAB.Team2Ban2 <> 'None' AND PAB.Team2Ban2 <> 'Missing Data' "
          "AND PAB.Team2Ban3 IS NOT NULL AND PAB.Team2Ban3 <> 'None' AND PAB.Team2Ban3 <> 'Missing Data' "
          "AND PAB.Team2Ban4 IS NOT NULL AND PAB.Team2Ban4 <> 'None' AND PAB.Team2Ban4 <> 'Missing Data' "
          "AND PAB.Team2Ban5 IS NOT NULL AND PAB.Team2Ban5 <> 'None' AND PAB.Team2Ban5 <> 'Missing Data' "
          ""
          "AND PAB.Team1Pick1 IS NOT NULL AND PAB.Team1Pick1 <> 'None' AND PAB.Team1Pick1 <> 'Missing Data' "
          "AND PAB.Team1Pick2 IS NOT NULL AND PAB.Team1Pick2 <> 'None' AND PAB.Team1Pick2 <> 'Missing Data' "
          "AND PAB.Team1Pick3 IS NOT NULL AND PAB.Team1Pick3 <> 'None' AND PAB.Team1Pick3 <> 'Missing Data' "
          "AND PAB.Team1Pick4 IS NOT NULL AND PAB.Team1Pick4 <> 'None' AND PAB.Team1Pick4 <> 'Missing Data' "
          "AND PAB.Team1Pick5 IS NOT NULL AND PAB.Team1Pick5 <> 'None' AND PAB.Team1Pick5 <> 'Missing Data' "
          ""
          "AND PAB.Team2Pick1 IS NOT NULL AND PAB.Team2Pick1 <> 'None' AND PAB.Team2Pick1 <> 'Missing Data' "
          "AND PAB.Team2Pick2 IS NOT NULL AND PAB.Team2Pick2 <> 'None' AND PAB.Team2Pick2 <> 'Missing Data' "
          "AND PAB.Team2Pick3 IS NOT NULL AND PAB.Team2Pick3 <> 'None' AND PAB.Team2Pick3 <> 'Missing Data' "
          "AND PAB.Team2Pick4 IS NOT NULL AND PAB.Team2Pick4 <> 'None' AND PAB.Team2Pick4 <> 'Missing Data' "
          "AND PAB.Team2Pick5 IS NOT NULL AND PAB.Team2Pick5 <> 'None' AND PAB.Team2Pick5 <> 'Missing Data' "
          f"AND (MS.Team1 IN {teams_filter} OR MS.Team2 IN {teams_filter})"
)

# Exemple de mapping : colonne originale -> nouveau nom
column_rename = {
    "Team1": "BLUE_TEAM",
    "Team2": "RED_TEAM",
    "Patch": "PATCH",
    "Team1Ban1": "BLUE_BAN1",
    "Team2Ban1": "RED_BAN1",
    "Team1Ban2": "BLUE_BAN2",
    "Team2Ban2": "RED_BAN2",
    "Team1Ban3": "BLUE_BAN3",
    "Team2Ban3": "RED_BAN3",
    "Team1Pick1": "BLUE_PICK1",
    "Team2Pick1": "RED_PICK1",
    "Team2Pick2": "RED_PICK2",
    "Team1Pick2": "BLUE_PICK2",
    "Team1Pick3": "BLUE_PICK3",
    "Team2Pick3": "RED_PICK3",
    "Team2Ban4": "RED_BAN4",
    "Team1Ban4": "BLUE_BAN4",
    "Team2Ban5": "RED_BAN5",
    "Team1Ban5": "BLUE_BAN5",
    "Team2Pick4": "RED_PICK4",
    "Team1Pick4": "BLUE_PICK4",
    "Team1Pick5": "BLUE_PICK5",
    "Team2Pick5": "RED_PICK5"
}

new_fieldnames = [column_rename.get(col, col) for col in drafts[0].keys()]

renamed_drafts = [{column_rename.get(k, k): v for k, v in row.items()} for row in drafts]

with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(renamed_drafts)

print(f"Dataset saved to {output_file}, total drafts: {len(drafts)}")

# -------- Patches --------
patches = sorted({row["Patch"] for row in drafts if row["Patch"]})
with (vocab_dir / "patches.txt").open("w", encoding="utf-8") as f:
    f.write("\n".join(patches))
print(f"{len(patches)} patches saved to vocab/patches.txt")
