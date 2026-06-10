import json

files = [
    "eval-reports/2026-06-10_13-23-38.json",
    "eval-reports/WINNER: 2026-06-09_16-07-17.json",
    "eval-reports/WINNER: 2026-06-09_15-36-52.json",
    "eval-reports/WINNER: 2026-06-09_22-17-30.json",
    "eval-reports/2026-06-09_23-50-47.json",
    "eval-reports/2026-06-10_09-43-06.json",
    "eval-reports/2026-06-10_14-32-07.json",
    "eval-reports/2026-06-10_09-59-34.json",
    "eval-reports/2026-06-09_23-34-45.json",
    "eval-reports/2026-06-09_23-47-23.json"
]

all_queries = []
all_items = []
base_options = None

for f in files:
    with open(f, "r") as file:
        data = json.load(file)
        if base_options is None:
            base_options = data.get("options", {})
        
        # Add queries
        for q in data.get("queries", []):
            all_queries.append(q)
            
        # Add items
        for item in data.get("items", []):
            all_items.append(item)

# Recompute summary
providers = ["tavily_search_advanced", "research_mini_lite", "tavily_research_mini"]
latency_sum = {p: 0 for p in providers}
latency_count = {p: 0 for p in providers}
quality_sum = {p: 0 for p in providers}
quality_count = {p: 0 for p in providers}
winners = {p: 0 for p in providers}
winners["tie"] = 0

for item in all_items:
    quality = item.get("quality", {})
    scores = quality.get("scores", {})
    winner = quality.get("winner")
    
    if winner:
        if winner in winners:
            winners[winner] += 1
        elif winner == "tie":
            winners["tie"] += 1
            
    for res in item.get("results", []):
        p = res["provider"]
        lat = res.get("latency_seconds")
        if lat is not None and lat > 1.0:
            latency_sum[p] += lat
            latency_count[p] += 1
            
        qual = scores.get(p, {}).get("overall")
        if qual is not None:
            quality_sum[p] += qual
            quality_count[p] += 1

summary = {
    "query_count": len(all_items),
    "provider_count": len(providers),
    "average_latency_seconds": {p: (latency_sum[p]/latency_count[p] if latency_count[p] > 0 else 0) for p in providers},
    "average_quality_overall": {p: (quality_sum[p]/quality_count[p] if quality_count[p] > 0 else 0) for p in providers},
    "winners": winners
}

big_eval = {
    "queries": all_queries,
    "options": base_options,
    "items": all_items,
    "summary": summary
}

with open("eval-reports/big_eval.json", "w") as out:
    json.dump(big_eval, out, indent=2)

print(f"Created big_eval.json with {len(all_items)} runs.")
