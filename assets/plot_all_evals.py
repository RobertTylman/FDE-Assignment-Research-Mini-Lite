import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import pi

files = ["eval-reports/big_eval.json"]

# Data structures
scatter_data = []
radar_data_raw = []
source_data = []
win_counts = {
    "tavily_search_advanced": 0,
    "research_mini_lite": 0,
    "tavily_research_mini": 0
}

# Color palette aligned with README
palette = {
    "tavily_search_advanced": "#38bdf8",   # Sky Blue
    "research_mini_lite": "#ec4899",       # Hot Pink
    "tavily_research_mini": "#a855f7"      # Purple
}
labels = {
    "tavily_search_advanced": "Search Advanced",
    "research_mini_lite": "Research Mini Lite",
    "tavily_research_mini": "Research Mini"
}

for f in files:
    with open(f, 'r') as file:
        data = json.load(file)
        
        for item in data.get('items', []):
            quality = item.get('quality') or {}
            scores = quality.get('scores', {})
            
            # Winner data
            winner = quality.get('winner')
            if winner in win_counts:
                win_counts[winner] += 1
            
            for res in item.get('results', []):
                provider = res.get('provider')
                latency = res.get('latency_seconds')
                source_count = res.get('source_count')
                
                if not provider or latency is None or latency <= 1.0:
                    continue
                    
                prov_scores = scores.get(provider, {})
                overall = prov_scores.get('overall')
                
                if overall is not None:
                    scatter_data.append({
                        "Model": labels.get(provider, provider),
                        "Provider_ID": provider,
                        "Latency (s)": latency,
                        "Overall Quality": overall
                    })
                
                if source_count is not None:
                    source_data.append({
                        "Model": labels.get(provider, provider),
                        "Source Count": source_count
                    })
                
                if prov_scores:
                    radar_data_raw.append({
                        "Provider": provider,
                        "Overall": prov_scores.get('overall', 0),
                        "Latency Score": prov_scores.get('latency', 0),
                        "Completeness": prov_scores.get('completeness', 0),
                        "Grounding": prov_scores.get('grounding', 0),
                        "Synthesis": prov_scores.get('synthesis', 0),
                        "Clarity": prov_scores.get('clarity', 0)
                    })

sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#cbd5e1", "grid.color": "#f1f5f9"})

# 1. Latency vs Quality Scatter Plot
df_scatter = pd.DataFrame(scatter_data)
if not df_scatter.empty:
    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(
        data=df_scatter, 
        x="Latency (s)", 
        y="Overall Quality", 
        hue="Provider_ID",
        palette=palette,
        s=120, alpha=0.7, edgecolor='white', linewidth=1.5
    )
    # Customize legend
    handles, legend_labels = ax.get_legend_handles_labels()
    new_labels = [labels.get(lbl, lbl) for lbl in legend_labels]
    ax.legend(handles, new_labels, title="Model", loc="lower right", frameon=True)
    
    plt.title("Latency vs. Overall Quality (The Sweet Spot)", fontsize=16, pad=15, fontweight='bold', color="#0f172a")
    plt.xlabel("Latency (Seconds)", fontsize=12, fontweight='bold', color="#334155")
    plt.ylabel("Overall Quality (1-5)", fontsize=12, fontweight='bold', color="#334155")
    plt.ylim(0.5, 5.5)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("assets/latency_vs_quality.png", dpi=300)
    plt.close()

# 2. Multi-Metric Quality Radar (Spider Chart)
df_radar_raw = pd.DataFrame(radar_data_raw)
if not df_radar_raw.empty:
    # Group by provider and compute mean
    df_radar_mean = df_radar_raw.groupby("Provider").mean().reset_index()
    categories = ['Overall', 'Latency Score', 'Completeness', 'Grounding', 'Synthesis', 'Clarity']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 12})
    
    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, color="#334155", size=12, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([1,2,3,4,5], ["1","2","3","4","5"], color="grey", size=10)
    plt.ylim(0,5.5)
    
    # Remove the outer spine for a cleaner look
    ax.spines['polar'].set_visible(False)
    
    for _, row in df_radar_mean.iterrows():
        prov = row['Provider']
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        color = palette.get(prov, "#000000")
        label = labels.get(prov, prov)
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', marker='o', markersize=6, label=label, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)
        
    plt.title("Multi-Metric Quality Averages", size=18, fontweight='bold', y=1.08, color="#0f172a")
    
    # Place legend at the bottom center to keep the radar chart perfectly centered
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=True, ncol=3)
    
    # Adjust layout to remove excessive whitespace while fitting the legend
    plt.tight_layout(pad=2.0)
    plt.savefig("assets/quality_radar.png", dpi=300)
    plt.close()

# 3. Source Density / Depth (Boxplot)
df_source = pd.DataFrame(source_data)
if not df_source.empty:
    plt.figure(figsize=(8, 8))
    
    reverse_labels = {v: k for k, v in labels.items()}
    
    sns.boxplot(
        data=df_source, 
        x="Model", 
        y="Source Count", 
        hue="Model",
        palette={lbl: palette[reverse_labels[lbl]] for lbl in labels.values()},
        width=0.4, boxprops={'alpha': 0.3}, showfliers=False, legend=False
    )
    sns.stripplot(
        data=df_source, 
        x="Model", 
        y="Source Count", 
        hue="Model",
        palette={lbl: palette[reverse_labels[lbl]] for lbl in labels.values()},
        size=6, alpha=0.7, jitter=True, edgecolor='white', linewidth=1, legend=False
    )
    
    plt.title("Source Density per Query", fontsize=16, pad=15, fontweight='bold', color="#0f172a")
    plt.xlabel("")
    plt.ylabel("Number of Unique Sources", fontsize=12, fontweight='bold', color="#334155")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("assets/source_density.png", dpi=300)
    plt.close()

# 4. Overall Win Rate (Donut Chart)
total_wins = sum(win_counts.values())
if total_wins > 0:
    plt.figure(figsize=(8, 8))
    
    # Filter out 0 wins
    plot_data = {k: v for k, v in win_counts.items() if v > 0}
    colors = [palette[k] for k in plot_data.keys()]
    chart_labels = [f"{labels[k]} ({v})" for k, v in plot_data.items()]
    
    plt.pie(
        plot_data.values(), 
        labels=chart_labels, 
        colors=colors,
        autopct='%1.1f%%', 
        startangle=90, 
        pctdistance=0.85,
        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': '#0f172a'},
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )
    
    plt.title("GPT-4 Judge: Overall Win Rate", fontsize=16, pad=15, fontweight='bold', color="#0f172a")
    plt.figtext(0.5, 0.03, 
                "*Disclaimer: LLM judges often introduce bias. For this eval, grading was:\n25% Completeness | 25% Latency | 20% Grounding | 15% Clarity | 10% Synthesis | 5% Sources", 
                ha="center", fontsize=9, color="black")
    plt.tight_layout()
    plt.savefig("assets/win_rate.png", dpi=300)
    plt.close()

print("Saved all 4 charts to assets/")
