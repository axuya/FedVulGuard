#!/usr/bin/env python3
import os
import re
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

SLITHER_ROOT = Path("results/sota_tools/slither")
SMARTCHECK_ROOT = Path("results/sota_tools/smartcheck")
SMARTCHECK_SUMMARY = Path("results/sota_tools/smartcheck_summary.csv")  # optional
OUT_CSV = Path("results/sota_tools/sota_tools_table.csv")

STRICT_IMPACT = {"High", "Medium"}
BROAD_IMPACT  = {"High", "Medium", "Low"}

# SmartCheck: typically ends with: ✖ 24 problems (24 errors)
PROB_PAT  = re.compile(r"✖\s*(\d+)\s*problems", re.IGNORECASE)
SEV_PAT   = re.compile(r"severity\s*:\s*([0-9]+)", re.IGNORECASE)
TIMEOUT_PAT = re.compile(r"TIMEOUT|timed out", re.IGNORECASE)
ERROR_PAT   = re.compile(r"error|exception|traceback|failed|cannot|parser", re.IGNORECASE)

def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def slither_stats(chain_dir: Path):
    n = 0
    ok = 0
    strict_hit = 0
    broad_hit = 0

    for p in chain_dir.glob("*.json"):
        n += 1
        js = read_json(p)
        if not js or not js.get("success", False):
            continue
        ok += 1
        detectors = js.get("results", {}).get("detectors", []) or []
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

    return {
        "slither_n": n,
        "slither_ok": ok,
        "slither_strict_hit": strict_hit,
        "slither_broad_hit": broad_hit,
        "slither_strict_rate": (strict_hit / n) if n else 0.0,
        "slither_broad_rate": (broad_hit / n) if n else 0.0,
    }

def parse_smartcheck_out(out_txt: str, err_txt: str = ""):
    all_txt = (out_txt or "") + "\n" + (err_txt or "")
    status = "ok"
    if TIMEOUT_PAT.search(all_txt):
        status = "timeout"
    elif ERROR_PAT.search(err_txt or "") and (err_txt or "").strip():
        status = "error"

    # problems count if present; else fallback to count severity lines
    m = PROB_PAT.search(out_txt or "")
    if m:
        findings = int(m.group(1))
    else:
        findings = len(SEV_PAT.findall(out_txt or ""))

    sev_cnt = Counter()
    for s in SEV_PAT.findall(out_txt or ""):
        try:
            sev_cnt[int(s)] += 1
        except Exception:
            pass

    hit = 1 if findings > 0 else 0
    return status, hit, findings, sev_cnt

def smartcheck_stats_from_outputs(chain_dir: Path):
    n = 0
    ok = 0
    timeout = 0
    error = 0
    hit = 0
    finding_sum = 0
    sev = Counter()

    for p in chain_dir.glob("*.out.txt"):
        n += 1
        out_txt = p.read_text(encoding="utf-8", errors="ignore")
        err_p = p.with_suffix(".err.txt")
        err_txt = err_p.read_text(encoding="utf-8", errors="ignore") if err_p.exists() else ""
        status, h, findings, sev_cnt = parse_smartcheck_out(out_txt, err_txt)

        if status == "ok":
            ok += 1
        elif status == "timeout":
            timeout += 1
        else:
            error += 1

        hit += h
        finding_sum += findings
        sev.update(sev_cnt)

    hit_rate = (hit / n) if n else 0.0
    finding_avg = (finding_sum / n) if n else 0.0  # avg per contract (more stable than per-hit)
    return {
        "smartcheck_n": n,
        "smartcheck_ok": ok,
        "smartcheck_timeout": timeout,
        "smartcheck_error": error,
        "smartcheck_hit": hit,
        "smartcheck_hit_rate": hit_rate,
        "smartcheck_finding_avg": finding_avg,
        "smartcheck_sev0": sev.get(0, 0),
        "smartcheck_sev1": sev.get(1, 0),
        "smartcheck_sev2": sev.get(2, 0),
        "smartcheck_sev3": sev.get(3, 0),
    }

def smartcheck_stats_from_summary_csv(summary_csv: Path):
    # Expect columns: tool,chain,n,ok,timeout,error,hit,hit_rate,finding_avg,sev0,sev1,sev2,sev3
    d = {}
    with summary_csv.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            chain = row.get("chain", "")
            if not chain:
                continue
            # Normalize types
            def to_int(x): 
                try: return int(float(x))
                except: return 0
            def to_f(x):
                try: return float(x)
                except: return 0.0

            d[chain] = {
                "smartcheck_n": to_int(row.get("n", 0)),
                "smartcheck_ok": to_int(row.get("ok", 0)),
                "smartcheck_timeout": to_int(row.get("timeout", 0)),
                "smartcheck_error": to_int(row.get("error", 0)),
                "smartcheck_hit": to_int(row.get("hit", 0)),
                "smartcheck_hit_rate": to_f(row.get("hit_rate", 0.0)),
                "smartcheck_finding_avg": to_f(row.get("finding_avg", 0.0)),
                "smartcheck_sev0": to_int(row.get("sev0", 0)),
                "smartcheck_sev1": to_int(row.get("sev1", 0)),
                "smartcheck_sev2": to_int(row.get("sev2", 0)),
                "smartcheck_sev3": to_int(row.get("sev3", 0)),
            }
    return d

def main():
    if not SLITHER_ROOT.exists():
        raise SystemExit(f"[ERR] missing {SLITHER_ROOT}")
    if not SMARTCHECK_ROOT.exists():
        raise SystemExit(f"[ERR] missing {SMARTCHECK_ROOT}")

    # Discover chains by slither dirs
    chains = sorted([p.name for p in SLITHER_ROOT.iterdir() if p.is_dir()])

    # SmartCheck stats
    sc_map = {}
    if SMARTCHECK_SUMMARY.exists():
        sc_map = smartcheck_stats_from_summary_csv(SMARTCHECK_SUMMARY)

    rows = []
    for chain in chains:
        sl_dir = SLITHER_ROOT / chain
        sc_dir = SMARTCHECK_ROOT / chain

        sl = slither_stats(sl_dir)
        if chain in sc_map:
            sc = sc_map[chain]
        else:
            if not sc_dir.exists():
                # allow missing chain if not run
                sc = {k: 0 for k in [
                    "smartcheck_n","smartcheck_ok","smartcheck_timeout","smartcheck_error",
                    "smartcheck_hit","smartcheck_hit_rate","smartcheck_finding_avg",
                    "smartcheck_sev0","smartcheck_sev1","smartcheck_sev2","smartcheck_sev3"
                ]}
            else:
                sc = smartcheck_stats_from_outputs(sc_dir)

        # Use the slither_n as primary n (since you enforce TSV=500)
        n = sl["slither_n"]
        row = {
            "chain": chain,
            "n": n,

            "slither_ok": sl["slither_ok"],
            "slither_strict_hit": sl["slither_strict_hit"],
            "slither_strict_rate": round(sl["slither_strict_rate"], 4),
            "slither_broad_hit": sl["slither_broad_hit"],
            "slither_broad_rate": round(sl["slither_broad_rate"], 4),

            "smartcheck_n": sc["smartcheck_n"],
            "smartcheck_ok": sc["smartcheck_ok"],
            "smartcheck_timeout": sc["smartcheck_timeout"],
            "smartcheck_error": sc["smartcheck_error"],
            "smartcheck_hit": sc["smartcheck_hit"],
            "smartcheck_hit_rate": round(sc["smartcheck_hit_rate"], 4),
            "smartcheck_finding_avg": round(sc["smartcheck_finding_avg"], 3),

            "smartcheck_sev0": sc.get("smartcheck_sev0", 0),
            "smartcheck_sev1": sc.get("smartcheck_sev1", 0),
            "smartcheck_sev2": sc.get("smartcheck_sev2", 0),
            "smartcheck_sev3": sc.get("smartcheck_sev3", 0),
        }
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote {OUT_CSV}")
    for r in rows:
        print(
            f"{r['chain']:<12} n={r['n']:4d} "
            f"slither_strict={r['slither_strict_hit']:4d}({r['slither_strict_rate']:.3f}) "
            f"slither_broad={r['slither_broad_hit']:4d}({r['slither_broad_rate']:.3f}) "
            f"smartcheck_hit={r['smartcheck_hit']:4d}({r['smartcheck_hit_rate']:.3f}) "
            f"smartcheck_finding_avg={r['smartcheck_finding_avg']:.2f}"
        )

if __name__ == "__main__":
    main()
