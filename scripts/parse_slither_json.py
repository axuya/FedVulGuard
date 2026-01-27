#解析Sliter Json的输出
# -*- coding: utf-8 -*-

import json
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

def safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(errors="ignore"))
    except Exception:
        return None

def parse_one(doc: dict):
    """
    Returns:
      success: bool
      error: str|None
      detector_count: int
      detector_names: list[str]
      finding_count: int   # total number of elements across detectors
    """
    if not isinstance(doc, dict):
        return False, "json_parse_fail", 0, [], 0

    success = bool(doc.get("success", False))
    error = doc.get("error", None)

    results = doc.get("results") or {}
    detectors = results.get("detectors") or []

    detector_names = []
    finding_count = 0

    # Slither JSON: detectors[] each has "elements": [...]
    for d in detectors:
        if not isinstance(d, dict):
            continue
        # name field can be "check" or "detector" depending on wrapper;
        # keep robust:
        name = d.get("check") or d.get("detector") or d.get("name") or "unknown_detector"
        detector_names.append(str(name))
        elems = d.get("elements") or []
        if isinstance(elems, list):
            finding_count += len(elems)

    detector_count = len(detector_names)
    return success, error, detector_count, detector_names, finding_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results/sota_tools/slither")
    ap.add_argument("--out", default="results/sota_tools/slither_parsed.csv")
    ap.add_argument("--out_summary", default="results/sota_tools/slither_summary_by_chain.csv")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out_summary = Path(args.out_summary)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    chain_stats = defaultdict(Counter)

    # expect subdirs: BSC_500, Fantom_500, ...
    for chain_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        chain = chain_dir.name
        for fp in sorted([p for p in chain_dir.iterdir() if p.is_file()]):
            doc = safe_load_json(fp)

            success, error, detector_count, detector_names, finding_count = parse_one(doc)

            # classify status
            if doc is None:
                status = "json_parse_fail"
            elif success is True:
                status = "ok"
            else:
                status = "fail"

            # hit definition: ok + at least one detector
            hit = 1 if (status == "ok" and detector_count > 0) else 0

            rows.append({
                "chain": chain,
                "file": fp.name,
                "status": status,
                "success": int(bool(success)),
                "error": "" if error is None else str(error),
                "detector_count": detector_count,
                "finding_count": finding_count,
                "detectors": "|".join(detector_names),
                "hit": hit,
                "bytes": fp.stat().st_size,
            })

            chain_stats[chain]["n_files"] += 1
            chain_stats[chain][f"status_{status}"] += 1
            chain_stats[chain]["hit"] += hit
            chain_stats[chain]["no_hit"] += (1 - hit)
            if status == "ok":
                chain_stats[chain]["ok"] += 1
            else:
                chain_stats[chain]["fail"] += 1

    # write per-contract parsed csv
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["chain","file","status","success","error","detector_count","finding_count","detectors","hit","bytes"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # write per-chain summary csv
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["chain","n_files","ok","fail","hit","no_hit","status_ok","status_fail","status_json_parse_fail"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for chain in sorted(chain_stats.keys()):
            c = chain_stats[chain]
            w.writerow({
                "chain": chain,
                "n_files": c.get("n_files", 0),
                "ok": c.get("ok", 0),
                "fail": c.get("fail", 0),
                "hit": c.get("hit", 0),
                "no_hit": c.get("no_hit", 0),
                "status_ok": c.get("status_ok", 0),
                "status_fail": c.get("status_fail", 0),
                "status_json_parse_fail": c.get("status_json_parse_fail", 0),
            })

    print(f"[OK] wrote: {out}  (rows={len(rows)})")
    print(f"[OK] wrote: {out_summary}")
    print("\n[Quick view]")
    for chain in sorted(chain_stats.keys()):
        c = chain_stats[chain]
        print(f"{chain:12s} n={c.get('n_files',0):4d} ok={c.get('ok',0):4d} hit={c.get('hit',0):4d} fail={c.get('fail',0):4d}")

if __name__ == "__main__":
    main()
