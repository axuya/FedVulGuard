#Slither detector 分布
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

def load_json(p: Path):
    try:
        return json.loads(p.read_text(errors="ignore"))
    except Exception:
        return None

def get_detectors(doc: dict):
    if not isinstance(doc, dict):
        return []
    res = doc.get("results") or {}
    dets = res.get("detectors") or []
    names = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        name = d.get("check") or d.get("detector") or d.get("name") or "unknown_detector"
        names.append(str(name))
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results/sota_tools/slither")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.root)
    global_cnt = Counter()
    per_chain = defaultdict(Counter)

    for chain_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        chain = chain_dir.name
        for fp in sorted([p for p in chain_dir.iterdir() if p.is_file()]):
            doc = load_json(fp)
            if not doc or not doc.get("success", False):
                continue
            for name in get_detectors(doc):
                global_cnt[name] += 1
                per_chain[chain][name] += 1

    print("\n[Top detectors - global]")
    for name, c in global_cnt.most_common(args.topk):
        print(f"{c:6d}  {name}")

    print("\n[Top detectors - per chain]")
    for chain in sorted(per_chain.keys()):
        print(f"\n== {chain} ==")
        for name, c in per_chain[chain].most_common(15):
            print(f"{c:6d}  {name}")

if __name__ == "__main__":
    main()
