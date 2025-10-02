from pathlib import Path

from mwrogue.esports_client import EsportsClient

vocab_dir = Path("../../resources/vocab")
vocab_dir.mkdir(exist_ok=True)

# --- Charger champions depuis wiki ---
site = EsportsClient("lol")
champions = site.cargo_client.query(
    tables="Champions=C",
    fields="C.Name"
)

champion_list = sorted({row["Name"] for row in champions if row["Name"]})

# --- Charger drafts locales ---
with (vocab_dir / "champions.txt").open("w", encoding="utf-8") as f:
    f.write("\n".join(champion_list))
print(f"{len(champion_list)} champions saved to vocab/champions.txt")