import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("../../resources/data/drafts_context_tokens.csv")

# Columns containing champions
champion_cols = [
    "BLUE_BAN1","RED_BAN1","BLUE_BAN2","RED_BAN2","BLUE_BAN3","RED_BAN3",
    "BLUE_PICK1","RED_PICK1","RED_PICK2","BLUE_PICK2","BLUE_PICK3","RED_PICK3",
    "RED_BAN4","BLUE_BAN4","RED_BAN5","BLUE_BAN5","RED_PICK4","BLUE_PICK4","BLUE_PICK5","RED_PICK5"
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
