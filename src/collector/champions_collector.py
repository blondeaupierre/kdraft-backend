import csv
import pandas as pd
from mwrogue.esports_client import EsportsClient

# --- Charger champions depuis wiki ---
site = EsportsClient("lol")
champions = site.cargo_client.query(
    tables="Champions=C",
    fields="C.Name"
)

# --- Charger drafts locales ---
df = pd.read_csv("../../resources/data/drafts.csv")
num_drafts = len(df)

# Colonnes de bans et picks
ban_cols = [c for c in df.columns if "Ban" in c]
pick_cols = [c for c in df.columns if "Pick" in c]

# Comptage brut
ban_counts = df[ban_cols].stack().value_counts()
pick_counts = df[pick_cols].stack().value_counts()

# --- Écriture du fichier final ---
with open("../../resources/data/champions.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["champion", "id", "ban_rate", "pick_rate"])

    for idx, champ in enumerate(champions):  # index de 0 à N
        name = champ["Name"]

        # calcul des stats
        total_bans = ban_counts.get(name, 0)
        total_picks = pick_counts.get(name, 0)

        ban_rate = total_bans / num_drafts
        pick_rate = total_picks / num_drafts

        writer.writerow([
            name,
            idx,
            f"{ban_rate:.4f}",
            f"{pick_rate:.4f}"
        ])
