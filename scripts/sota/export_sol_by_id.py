# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = []

    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            sid = o["id"]
            src = o.get("src_path")
            if not src:
                raise RuntimeError("Missing src_path in jsonl record. Expect keys: id, src_path, label.")
            p = Path(src)
            if not p.is_absolute():
                p = (root / p).resolve()
            rows.append(f"{sid}\t{p}")

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[OK] wrote {len(rows)} entries to {out}")

if __name__ == "__main__":
    main()
