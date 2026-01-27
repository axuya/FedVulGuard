#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict

IMPACT_OK = {"High", "Medium"}
CONF_OK   = {"High", "Medium"}

# STRICT: keep only "hard" security findings to avoid near-100% positives
STRICT_ALLOW = {
    "arbitrary-send-eth",
    "unchecked-transfer",
    "locked-ether",
    "weak-prng",
    "uninitialized-state",
    "uninitialized-local",
    "encode-packed-collision",
    "return-bomb",
}

def load_json(p: Path):
    try:
        return json.loads(p.read_text(errors="ignore"))
    except Exception:
        return None

def load_allowlist(p: Path):
    if not p.exists():
        return set()
    s = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            s.add(t)
    return s

def norm_level(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip().capitalize()
    return str(x).strip().capitalize()

def parse_detectors(doc: dict):
    res = doc.get("results") or {}
    dets = res.get("detectors") or []
    out = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        name = d.get("check") or d.get("detector") or d.get("name") or "unknown_detector"
        impact = norm_level(d.get("impact"))
        conf = norm_level(d.get("confidence"))
        elems = d.get("elements") or []
        n_elem = len(elems) if isinstance(elems, list) else 0
        out.append((str(name), impact, conf, n_elem))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results/sota_tools/slither")
    ap.add_argument("--broad_allowlist", default="configs/slither_vuln_detectors.txt")
    ap.add_argument("--out", default="results/sota_tools/slither_tiered.csv")
    ap.add_argument("--out_summary", default="results/sota_tools/slither_tiered_summary.csv")
    args = ap.parse_args()

    root = Path(args.root)
    broad_allow = load_allowlist(Path(args.broad_allowlist))  # your broader vuln list

    rows = []
    chain_stats = defaultdict(Counter)

    for chain_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        chain = chain_dir.name
        for fp in sorted([p for p in chain_dir.iterdir() if p.is_file()]):
            doc = load_json(fp)
            if doc is None:
                status = "json_parse_fail"
                success = False
                det_info = []
            else:
                success = bool(doc.get("success", False))
                status = "ok" if success else "fail"
                det_info = parse_detectors(doc) if success else []

            strict_find = 0
            broad_find = 0

            for name, impact, conf, n_elem in det_info:
                impact_ok = (impact in IMPACT_OK) if impact is not None else True
                conf_ok   = (conf in CONF_OK) if conf is not None else True
                if not (impact_ok and conf_ok):
                    continue

                if name in STRICT_ALLOW:
                    strict_find += n_elem
                if (not broad_allow) or (name in broad_allow):
                    broad_find += n_elem

            strict_hit = 1 if (status == "ok" and strict_find > 0) else 0
            broad_hit  = 1 if (status == "ok" and broad_find > 0) else 0

            rows.append({
                "chain": chain,
                "file": fp.name,
                "status": status,
                "success": int(success),
                "strict_findings": strict_find,
                "strict_hit": strict_hit,
                "broad_findings": broad_find,
                "broad_hit": broad_hit,
            })

            c = chain_stats[chain]
            c["n"] += 1
            c[f"status_{status}"] += 1
            c["strict_hit"] += strict_hit
            c["broad_hit"] += broad_hit

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    out2 = Path(args.out_summary); out2.parent.mkdir(parents=True, exist_ok=True)
    with out2.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["chain","n","status_ok","status_fail","status_json_parse_fail","strict_hit","broad_hit"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for chain in sorted(chain_stats.keys()):
            c = chain_stats[chain]
            w.writerow({
                "chain": chain,
                "n": c.get("n",0),
                "status_ok": c.get("status_ok",0),
                "status_fail": c.get("status_fail",0),
                "status_json_parse_fail": c.get("status_json_parse_fail",0),
                "strict_hit": c.get("strict_hit",0),
                "broad_hit": c.get("broad_hit",0),
            })

    print(f"[OK] wrote: {out}")
    print(f"[OK] wrote: {out2}")
    for chain in sorted(chain_stats.keys()):
        c = chain_stats[chain]
        print(f"{chain:12s} n={c.get('n',0):4d} strict_hit={c.get('strict_hit',0):4d} broad_hit={c.get('broad_hit',0):4d}")

if __name__ == "__main__":
    main()
