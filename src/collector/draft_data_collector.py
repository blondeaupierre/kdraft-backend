import csv

from mwrogue.esports_client import EsportsClient

# Définir le fichier de sortie
output_file = "../../resources/data/drafts_from_25_16.csv"

site = EsportsClient("lol")

top_4_teams = site.cargo_client.query(
    tables="TournamentResults=TR",
    fields="TR.Team",
    where="TR.Place_Number <= '4'"
          "AND TR.Tier = 'Offline'"
          # "AND TR.Tier = 'Offline' OR  TR.Tier = 'Online/Offline'"
          "AND TR.Date >= '2025-04-01'"
          "AND TR.Team IS NOT NULL",
    group_by="TR.Team"
)

teams = [row["Team"] for row in top_4_teams]

# échappe les apostrophes simples
escaped = [t.replace("'", "''") for t in teams if t is not None]

teams_filter = "('" + "','".join(escaped) + "')"

print("Number of teams : " + str(len(teams)))
print(teams_filter)

drafts = site.cargo_client.query(
    tables="MatchSchedule=MS, PicksAndBansS7=PAB",
    join_on="MS.MatchId=PAB.MatchId",
    fields="PAB.Team1Ban1, "
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
    where="MS.Patch>='25.16' "
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

fieldnames = drafts[0].keys()

# Écriture dans un fichier CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(drafts)

print(f"Dataset saved to {output_file}, total drafts: {len(drafts)}")