#!/usr/bin/env python3
import csv
import re
from pathlib import Path
from collections import defaultdict

# Inputs
TOOLS_CSV = Path("results/sota_tools/sota_tools_table.csv")  # 你前面合并出来的工具表
CGN_SUMMARY = Path("summary_all_runs_best.csv")              # 你上传的这个
OUT_CSV = Path("results/sota_tools/tool_vs_crossgraphnet.csv")

CHAIN_ORDER = ["Ethereum", "BSC", "Fantom", "Polygon"]
CHAIN_CANON = {"ethereum": "Ethereum", "bsc": "BSC", "fantom": "Fantom", "polygon": "Polygon"}

def chain_from_test_path(p: str):
    s = (p or "").lower()
    m = re.search(r"(ethereum|bsc|fantom|polygon)_500\.jsonl", s)
    if m:
        return CHAIN_CANON[m.group(1)]
    return None

def load_tools():
    if not TOOLS_CSV.exists():
        raise SystemExit(f"[ERR] missing tools csv: {TOOLS_CSV} (先跑 make_sota_tools_table.py)")
    tools = {}
    with TOOLS_CSV.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            chain = r["chain"]
            # chain 可能是 Ethereum_500 这种
            chain2 = chain.replace("_500","")
            tools[chain2] = r
    return tools

def load_crossgraphnet_metrics():
    if not CGN_SUMMARY.exists():
        raise SystemExit(f"[ERR] missing CrossGraphNet summary: {CGN_SUMMARY}")

    # 按 (chain, mode) 聚合多个 seed：mean F1/AUC
    agg = defaultdict(lambda: {"f1": [], "auc": []})

    with CGN_SUMMARY.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            mode = (r.get("mode") or "").strip().lower()  # stats / llm
            test_path = r.get("test_path") or ""
            chain = chain_from_test_path(test_path)
            if chain is None:
                continue

            # 只收 stats/llm 两类（你这份 CSV 就是它们）
            if mode not in ("stats", "llm"):
                continue

            try:
                f1 = float(r.get("best_f1", ""))
                auc = float(r.get("best_auc", ""))
            except Exception:
                continue

            agg[(chain, mode)]["f1"].append(f1)
            agg[(chain, mode)]["auc"].append(auc)

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    out = {}
    for chain in CHAIN_ORDER:
        out[chain] = {}
        for mode in ("stats", "llm"):
            f1m = mean(agg[(chain, mode)]["f1"])
            aucm = mean(agg[(chain, mode)]["auc"])
            out[chain][mode] = {
                "f1_mean": f1m,
                "auc_mean": aucm,
                "n": len(agg[(chain, mode)]["f1"]),
            }
        # best-of-two by AUC then F1
        cand = []
        for mode in ("stats", "llm"):
            a = out[chain][mode]["auc_mean"]
            f1m = out[chain][mode]["f1_mean"]
            if a is not None and f1m is not None:
                cand.append((a, f1m, mode))
        if cand:
            cand.sort(reverse=True)
            best_mode = cand[0][2]
        else:
            best_mode = None
        out[chain]["best_mode"] = best_mode
    return out

def f4(x):
    return "" if x is None else f"{x:.4f}"

def main():
    tools = load_tools()
    cgn = load_crossgraphnet_metrics()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "chain","n",
            "slither_strict_hit_rate","slither_broad_hit_rate","smartcheck_hit_rate",
            "crossgraphnet_stats_f1_mean","crossgraphnet_stats_auc_mean",
            "crossgraphnet_llm_f1_mean","crossgraphnet_llm_auc_mean",
            "crossgraphnet_best_mode","crossgraphnet_best_f1_mean","crossgraphnet_best_auc_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for chain in CHAIN_ORDER:
            tr = tools.get(chain, None)
            if tr:
                n = tr.get("n","")
                sl_strict = tr.get("slither_strict_rate","")
                sl_broad  = tr.get("slither_broad_rate","")
                sc_rate   = tr.get("smartcheck_hit_rate","")
            else:
                n, sl_strict, sl_broad, sc_rate = "", "", "", ""

            stats = cgn[chain]["stats"]
            llm   = cgn[chain]["llm"]
            best_mode = cgn[chain]["best_mode"]

            best_f1 = best_auc = None
            if best_mode:
                best_f1 = cgn[chain][best_mode]["f1_mean"]
                best_auc = cgn[chain][best_mode]["auc_mean"]

            w.writerow({
                "chain": chain,
                "n": n,
                "slither_strict_hit_rate": sl_strict,
                "slither_broad_hit_rate": sl_broad,
                "smartcheck_hit_rate": sc_rate,

                "crossgraphnet_stats_f1_mean": f4(stats["f1_mean"]),
                "crossgraphnet_stats_auc_mean": f4(stats["auc_mean"]),
                "crossgraphnet_llm_f1_mean": f4(llm["f1_mean"]),
                "crossgraphnet_llm_auc_mean": f4(llm["auc_mean"]),

                "crossgraphnet_best_mode": best_mode or "",
                "crossgraphnet_best_f1_mean": f4(best_f1),
                "crossgraphnet_best_auc_mean": f4(best_auc),
            })

    print(f"[OK] wrote {OUT_CSV}")
    print("[NOTE] Ethereum row will be blank for CrossGraphNet if you don't have tests on Ethereum_500 (e.g., *_to_Ethereum_500).")

if __name__ == "__main__":
    main()
