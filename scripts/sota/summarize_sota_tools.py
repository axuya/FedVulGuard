#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
import pandas as pd

LABEL_KEYS = ["label", "y", "vulnerable", "is_vuln", "target"]

def safe_name(rel: str) -> str:
    return re.sub(r"[/ ]", "_", rel)

def slither_pred(path: Path) -> int:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        res = obj.get("results", {})
        det = res.get("detectors") or res.get("findings") or []
        return 1 if isinstance(det, list) and len(det) > 0 else 0
    except Exception:
        return 0

def mythril_pred(path: Path) -> int:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        issues = obj.get("issues", [])
        return 1 if isinstance(issues, list) and len(issues) > 0 else 0
    except Exception:
        return 0

def prf(y_true, y_pred):
    tp = sum((t==1 and p==1) for t,p in zip(y_true,y_pred))
    fp = sum((t==0 and p==1) for t,p in zip(y_true,y_pred))
    fn = sum((t==1 and p==0) for t,p in zip(y_true,y_pred))
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return tp, fp, fn, prec, rec, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--tool_out_dir", default="results/sota_tools")
    ap.add_argument("--split_name", required=True)  # e.g., BSC_500
    ap.add_argument("--out_csv", default="results/sota_tools/sota_tools_summary.csv")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    test = Path(args.test_jsonl)
    out_root = Path(args.tool_out_dir)

    # load GT + src_path
    y_true, src_paths = [], []
    with test.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            y = None
            for k in LABEL_KEYS:
                if k in o:
                    y = o[k]; break
            if y is None:
                raise RuntimeError(f"Missing label in {test}. expected one of {LABEL_KEYS}")
            p = o.get("src_path") or o.get("sol_path") or o.get("path")
            if not p:
                raise RuntimeError(f"Missing src_path in {test}. expected src_path/sol_path/path")
            y_true.append(int(y))
            src_paths.append(p)

    slither_dir = out_root / "slither" / args.split_name
    mythril_dir = out_root / "mythril" / args.split_name

    y_sl, y_my = [], []
    missing_sl, missing_my = 0, 0
    for p in src_paths:
        p = Path(p)
        if not p.is_absolute():
            p = (root / p).resolve()
        rel = str(p).replace(str(root)+"/", "")
        fname = safe_name(rel) + ".json"

        sp = slither_dir / fname
        mp = mythril_dir / fname
        if not sp.exists(): missing_sl += 1
        if not mp.exists(): missing_my += 1

        y_sl.append(slither_pred(sp) if sp.exists() else 0)
        y_my.append(mythril_pred(mp) if mp.exists() else 0)

    rows = []
    for tool, y_pred, miss in [
        ("slither", y_sl, missing_sl),
        ("mythril", y_my, missing_my),
    ]:
        tp, fp, fn, prec, rec, f1 = prf(y_true, y_pred)
        rows.append({
            "split": args.split_name,
            "tool": tool,
            "n": len(y_true),
            "missing_reports": miss,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)

    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df_old = df_old[~((df_old["split"]==args.split_name) & (df_old["tool"].isin(df_new["tool"])))]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(out_csv, index=False)
    print("[OK] wrote:", out_csv)
    print(df_new.to_string(index=False))

if __name__ == "__main__":
    main()
