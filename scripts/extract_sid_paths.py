import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="path to *_500.jsonl")
    ap.add_argument("--out", required=True, help="output tsv path")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)

    assert jsonl_path.exists(), f"jsonl not found: {jsonl_path}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with jsonl_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = obj.get("sid") or obj.get("id")
            sol = obj.get("sol_path") or obj.get("path") or obj.get("file")

            if not sid or not sol:
                continue

            fout.write(f"{sid}\t{sol}\n")
            n += 1

    print(f"[OK] wrote {n} rows to {out_path}")

if __name__ == "__main__":
    main()