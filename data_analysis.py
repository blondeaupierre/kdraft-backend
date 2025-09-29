import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("resources/data/drafts.csv")

# Columns containing champions
champion_cols = [
    "Team1Ban1","Team2Ban1","Team1Ban2","Team2Ban2","Team1Ban3","Team2Ban3",
    "Team1Pick1","Team2Pick1","Team2Pick2","Team1Pick2","Team1Pick3","Team2Pick3",
    "Team2Ban4","Team1Ban4","Team2Ban5","Team1Ban5","Team2Pick4","Team1Pick4","Team1Pick5","Team2Pick5"
]

# Flatten all champion columns
all_champions = df[champion_cols].stack()

# Count frequency
champion_counts = all_champions.value_counts()
total_picks = champion_counts.sum()

# Add frequency %
champion_freq = (champion_counts / total_picks * 100).round(2)

# Print all champions with counts and %
print("Champion\tCount\tFreq (%)")
for champion in champion_counts.index:
    print(f"{champion}\t{champion_counts[champion]}\t{champion_freq[champion]}")

# Plot top N champions
top_n = 50
plt.figure(figsize=(10,6))
champion_counts.head(top_n).plot(kind='bar')
plt.title(f"Top {top_n} Most Common Champions")
plt.ylabel("Frequency")
plt.xlabel("Champion")
plt.xticks(rotation=90)
plt.show()
