"""Compare baselines vs. multi-agent modes on a labeled CSV (columns: text, category, sentiment).
Usage: python evaluate.py sample_data.csv [single vote full routed]"""
import sys, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pipeline import MODES

COST_PER_CALL = 0.0005  # rough placeholder USD per call - replace with your real pricing

path = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
modes = sys.argv[2:] or list(MODES)
df = pd.read_csv(path)
summary = []

for mode in modes:
    rows = [MODES[mode](t) for t in df["text"]]
    out = pd.DataFrame(rows)
    row = {"mode": mode}
    for dim in ("category", "sentiment"):
        row[f"{dim}_acc"] = accuracy_score(df[dim], out[dim])
        row[f"{dim}_macroF1"] = f1_score(df[dim], out[dim], average="macro", zero_division=0)
    row["human_review_%"] = 100 * (out["status"] == "human_review").mean()
    row["avg_calls"] = out["llm_calls"].mean()
    row["avg_latency_s"] = out["latency"].mean()
    row["est_cost_per_1000"] = row["avg_calls"] * COST_PER_CALL * 1000
    summary.append(row)
    print(f"\n=== {mode} ===")
    print(classification_report(df["category"], out["category"], zero_division=0))
    labels = sorted(set(df["category"]) | set(out["category"]))
    print(pd.DataFrame(confusion_matrix(df["category"], out["category"], labels=labels),
                       index=labels, columns=labels))

print("\n=== SUMMARY ===")
print(pd.DataFrame(summary).round(3).to_string(index=False))
