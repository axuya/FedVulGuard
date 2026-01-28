#FEDVUL vs SOTA
import csv
from collections import defaultdict

IN_CSV = "results/experiments/crosschain_runs/summary_all_runs_best.csv"
OUT_CSV = "results/sota_tools/crossgraphnet_recall_500.csv"

# we only care about these chains
CHAINS = {
    "Ethereum_500": "Ethereum",
    "BSC_500": "BSC",
    "Fantom_500": "Fantom",
    "Polygon_500": "Polygon",
}

METHOD_NAME = "CrossGraphNet"

def main():
    rows = []

    with open(IN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            run = r.get("run", "")
            mode = r.get("mode", "")
            model = r.get("model", "")
            recall = r.get("test_recall") or r.get("recall") or r.get("test_rec")

            if METHOD_NAME not in model:
                continue
            if run not in CHAINS:
                continue
            if not recall:
                continue

            rows.append({
                "chain": CHAINS[run],
                "run": run,
                "recall": float(recall),
            })

    # aggregate: take mean recall per chain (or max if you prefer)
    agg = defaultdict(list)
    for r in rows:
        agg[r["chain"]].append(r["recall"])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chain", "crossgraphnet_recall_mean"])
        for chain, vals in agg.items():
            writer.writerow([chain, round(sum(vals) / len(vals), 4)])

    print(f"[OK] wrote {OUT_CSV}")
    for chain, vals in agg.items():
        print(chain, "mean_recall=", round(sum(vals) / len(vals), 4))

if __name__ == "__main__":
    main()
