#!/usr/bin/env python3
import os
import json
import csv
from collections import defaultdict

# ----------------------------
# Config
# ----------------------------
SLITHER_ROOT = "results/sota_tools/slither"
SMARTCHECK_ROOT = "results/sota_tools/smartcheck"
OUT_CSV = "results/sota_tools/sota_tools_matrix.csv"

STRICT_IMPACT = {"High", "Medium"}
BROAD_IMPACT = {"High", "Medium", "Low"}


# ----------------------------
# Slither parsing
# ----------------------------
def parse_slither_chain(chain_dir):
    n = 0
    strict_hit = 0
    broad_hit = 0

    for fn in os.listdir(chain_dir):
        if not fn.endswith(".json"):
            continue
        n += 1
        path = os.path.join(chain_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue

        if not js.get("success", False):
            continue

        detectors = js.get("results", {}).get("detectors", [])
        strict = False
        broad = False

        for d in detectors:
            impact = d.get("impact", "")
            if impact in STRICT_IMPACT:
                strict = True
                broad = True
                break
            if impact in BROAD_IMPACT:
                broad = True

        if strict:
            strict_hit += 1
        if broad:
            broad_hit += 1

    return n, strict_hit, broad_hit


# ----------------------------
# SmartCheck parsing
# ----------------------------
def parse_smartcheck_chain(chain_dir):
    n = 0
    hit = 0
    total_findings = 0

    for fn in os.listdir(chain_dir):
        if not fn.endswith(".out.txt"):
            continue

        n += 1
        path = os.path.join(chain_dir, fn)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue

        # heuristic: any "severity:" line means hit
        findings = [
            ln for ln in lines
            if "severity:" in ln.lower()
        ]

        if findings:
            hit += 1
            total_findings += len(findings)

    hit_rate = hit / n if n > 0 else 0.0
    finding_avg = total_findings / hit if hit > 0 else 0.0

    return n, hit, hit_rate, finding_avg


# ----------------------------
# Main
# ----------------------------
def main():
    chains = sorted(
        d for d in os.listdir(SLITHER_ROOT)
        if os.path.isdir(os.path.join(SLITHER_ROOT, d))
    )

    rows = []

    for chain in chains:
        sl_dir = os.path.join(SLITHER_ROOT, chain)
        sc_dir = os.path.join(SMARTCHECK_ROOT, chain)

        if not os.path.isdir(sc_dir):
            print(f"[WARN] missing smartcheck dir for {chain}, skip")
            continue

        n_sl, strict_hit, broad_hit = parse_slither_chain(sl_dir)
        n_sc, sc_hit, sc_hit_rate, sc_finding_avg = parse_smartcheck_chain(sc_dir)

        n = min(n_sl, n_sc)

        rows.append({
            "chain": chain,
            "n": n,
            "slither_strict_hit": strict_hit,
            "slither_broad_hit": broad_hit,
            "smartcheck_hit": sc_hit,
            "smartcheck_hit_rate": round(sc_hit_rate, 4),
            "smartcheck_finding_avg": round(sc_finding_avg, 2),
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chain", "n",
                "slither_strict_hit", "slither_broad_hit",
                "smartcheck_hit", "smartcheck_hit_rate",
                "smartcheck_finding_avg"
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] wrote {OUT_CSV}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
