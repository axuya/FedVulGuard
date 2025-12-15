import json
import hashlib
from pathlib import Path

def make_graph_id(chain, src_path):
    raw = f"{chain}::{src_path}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def main():
    base = Path("data/graphs_cfg_norm")
    out_dir = Path("data/index")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg_file in base.glob("*.jsonl"):
        chain = cfg_file.stem
        index = {}

        with cfg_file.open("r", encoding="utf8") as f:
            for line in f:
                g = json.loads(line)

                nodes = g.get("cfg_nodes", [])
                if not nodes:
                    continue

                sm = nodes[0].get("source_mapping")
                if not sm:
                    continue

                src_path = sm.get("filename")
                if not src_path:
                    continue

                gid = make_graph_id(chain, src_path)
                index[src_path] = gid

        out_path = out_dir / f"{chain}_src2id.json"
        with out_path.open("w", encoding="utf8") as fout:
            json.dump(index, fout, indent=2)

        print(f"[OK] {chain}: {len(index)} entries -> {out_path}")

if __name__ == "__main__":
    main()
