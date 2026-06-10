import json
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

files = ["eval-reports/big_eval.json"]

data_records = []

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        for item in data.get('items', []):
            for res in item.get('results', []):
                provider = res.get('provider')
                latency = res.get('latency_seconds')
                # Filter out 0-second points (failed or cached runs)
                if provider and latency is not None and latency > 1.0:
                    data_records.append({
                        "Model": provider,
                        "Latency (s)": latency
                    })

df = pd.DataFrame(data_records)

# Map provider names to more readable labels
label_map = {
    "tavily_search_advanced": "Tavily Search Advanced",
    "research_mini_lite": "Research Mini Lite",
    "tavily_research_mini": "Tavily Research Mini"
}
df['Model'] = df['Model'].map(lambda x: label_map.get(x, x))

# Professional styling with playful colors
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(8, 8))

# Playful color palette with RML as the standout color
custom_palette = {
    "Tavily Search Advanced": "#75D2F6",    # Playful sky blue
    "Research Mini Lite": "#FF2A6D",        # Standout vibrant hot pink/red
    "Tavily Research Mini": "#B28DFF"       # Playful soft purple
}

model_order = ["Tavily Search Advanced", "Research Mini Lite", "Tavily Research Mini"]

sns.boxplot(
    data=df, 
    x="Model", 
    y="Latency (s)", 
    order=model_order,
    palette=custom_palette, 
    hue="Model", 
    legend=False, 
    width=0.5, 
    fliersize=0,
    boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1.5),
    medianprops=dict(color="white", linewidth=2)
)

sns.stripplot(
    data=df, 
    x="Model", 
    y="Latency (s)", 
    order=model_order,
    hue="Model",
    palette=custom_palette, 
    legend=False,
    size=5, 
    jitter=True, 
    alpha=0.6,
    edgecolor="black",
    linewidth=0.5
)

# Add average latency text above each group
means = df.groupby('Model')['Latency (s)'].mean()
for i, category in enumerate(model_order):
    if category in means:
        mean_val = means[category]
        # Place text slightly above the maximum value for this category
        max_val = df[df['Model'] == category]['Latency (s)'].max()
        # Add a tiny offset so it floats above the highest dot/whisker
        offset = df['Latency (s)'].max() * 0.05
        ax.text(i, max_val + offset, f"Avg: {mean_val:.2f}s", 
                horizontalalignment='center', size=11, color='black', weight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

sns.despine(left=True)

plt.title("Research Model Latency Distributions", fontsize=16, fontweight='bold', pad=25)
plt.xlabel("", fontsize=14)
plt.ylabel("Latency (seconds)", fontsize=14)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust y-limit so annotations aren't cut off
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.1)

plt.tight_layout()

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/latency_distribution.png", dpi=300)
print("Saved playful chart with averages to assets/latency_distribution.png")
