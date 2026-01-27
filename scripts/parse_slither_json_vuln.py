#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict

def load_json(p: Path):
    try:
        return json.loads(p.read_text(errors="ignore"))
    except Exception:
        return None

def load_allowlist(p: Path):
    if not p.exists():
        return None
    items = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            items.append(s)
    return set(items) if items else None

def parse_detectors(doc: dict):
    res = doc.get("results") or {}
    dets = res.get("detectors") or []
    names = []
    finding_count = 0
    for d in dets:
        if not isinstance(d, dict):
            continue
        name = d.get("check") or d.get("detector") or d.get("name") or "unknown_detector"
        names.append(str(name))
        elems = d.get("elements") or []
        if isinstance(elems, list):
            finding_count += len(elems)
    return names, finding_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--allowlist", default="configs/slither_vuln_detectors.txt")
    ap.add_argument("--out", default="results/sota_tools/slither_parsed_vuln.csv")
    ap.add_argument("--out_summary", default="results/sota_tools/slither_summary_vuln_by_chain.csv")
    args = ap.parse_args()

    root = Path(args.root)
    allow = load_allowlist(Path(args.allowlist))  # None => fallback to any-detector hit

    rows = []
    chain_stats = defaultdict(Counter)

    for chain_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        chain = chain_dir.name
        for fp in sorted([p for p in chain_dir.iterdir() if p.is_file()]):
            doc = load_json(fp)
            if doc is None:
                status = "json_parse_fail"
                success = False
                error = "json_parse_fail"
                det_names, finding_count = [], 0
            else:
                success = bool(doc.get("success", False))
                error = doc.get("error", None)
                status = "ok" if success else "fail"
                det_names, finding_count = parse_detectors(doc) if success else ([], 0)

            det_set = set(det_names)
            if status == "ok":
                if allow is None:
                    vuln_hit = 1 if len(det_names) > 0 else 0
                else:
                    vuln_hit = 1 if len(det_set.intersection(allow)) > 0 else 0
            else:
                vuln_hit = 0

            rows.append({
                "chain": chain,
                "file": fp.name,
                "status": status,
                "success": int(success),
                "error": "" if error is None else str(error),
                "detector_count": len(det_names),
                "finding_count": finding_count,
                "detectors": "|".join(det_names),
                "vuln_hit": vuln_hit,
            })

            c = chain_stats[chain]
            c["n_files"] += 1
            c[f"status_{status}"] += 1
            c["vuln_hit"] += vuln_hit
            c["vuln_clean"] += (1 - vuln_hit)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    out2 = Path(args.out_summary); out2.parent.mkdir(parents=True, exist_ok=True)
    with out2.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["chain","n_files","status_ok","status_fail","status_json_parse_fail","vuln_hit","vuln_clean"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for chain in sorted(chain_stats.keys()):
            c = chain_stats[chain]
            w.writerow({k: c.get(k, 0) if k!="chain" else chain for k in fieldnames})

    print(f"[OK] wrote: {out}")
    print(f"[OK] wrote: {out2}")
    for chain in sorted(chain_stats.keys()):
        c = chain_stats[chain]
        print(f"{chain:12s} n={c.get('n_files',0):4d} vuln_hit={c.get('vuln_hit',0):4d} ok={c.get('status_ok',0):4d}")

if __name__ == "__main__":
    main()
