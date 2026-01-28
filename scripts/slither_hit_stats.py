#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict

# ----------------------------
# Detector severity mapping
# ----------------------------
# Slither detectors are tagged with:
#   impact: High / Medium / Low / Informational
#
# We define:
#   strict_hit = impact in {High, Medium}
#   broad_hit  = impact in {High, Medium, Low}
#
STRICT_IMPACT = {"High", "Medium"}
BROAD_IMPACT  = {"High", "Medium", "Low"}


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def analyze_contract(js):
    """
    Analyze a single slither JSON output.
    Returns:
        (strict_hit: bool, broad_hit: bool)
    """
    if not js:
        return False, False

    if not js.get("success", False):
        return False, False

    detectors = js.get("results", {}).get("detectors", [])
    if not detectors:
        return False, False

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

    return strict, broad


def analyze_chain(chain_dir):
    """
    Analyze all slither outputs under one chain directory.
    """
    n = 0
    strict_hit = 0
    broad_hit = 0

    for fn in os.listdir(chain_dir):
        if not fn.endswith(".json"):
            continue

        n += 1
        path = os.path.join(chain_dir, fn)
        js = load_json(path)

        strict, broad = analyze_contract(js)
        if strict:
            strict_hit += 1
        if broad:
            broad_hit += 1

    return n, strict_hit, broad_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Path to results/sota_tools/slither"
    )
    args = ap.parse_args()

    root = args.root
    chains = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    for chain in chains:
        chain_dir = os.path.join(root, chain)
        n, strict_hit, broad_hit = analyze_chain(chain_dir)

        print(
            f"{chain:<12} "
            f"n={n:4d}  "
            f"strict_hit={strict_hit:4d}  "
            f"broad_hit={broad_hit:4d}"
        )


if __name__ == "__main__":
    main()
