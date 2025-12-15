import json
import hashlib
from pathlib import Path
from tqdm import tqdm

CHAINS = ["Ethereum", "BSC", "Polygon", "Avalanche", "Fantom"]

AST_DIR = Path("data/graphs_ast_llm")
CFG_DIR = Path("data/graphs_cfg_contract")
OUT_DIR = Path("data/train/crossgraphnet_lite")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_graph_id(chain: str, src_path: str) -> str:
    raw = f"{chain}::{src_path}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cfg(chain: str):
    """
    返回:
      src_path -> cfg_graph
    """
    cfg_map = {}
    path = CFG_DIR / f"{chain}.jsonl"
    with path.open("r", encoding="utf8") as f:
        for line in f:
            try:
                g = json.loads(line)
            except Exception:
                continue

            nodes = g.get("cfg_nodes", [])
            if not nodes:
                continue

            sm = nodes[0].get("source_mapping")
            if not sm:
                continue

            src_path = sm.get("filename")
            if not src_path:
                continue

            cfg_map[src_path] = {
                "cfg_nodes": g.get("cfg_nodes", []),
                "cfg_edges": g.get("cfg_edges", [])
            }
    return cfg_map


def main():
    for chain in CHAINS:
        ast_path = AST_DIR / f"{chain}.jsonl"
        cfg_path = CFG_DIR / f"{chain}.jsonl"
        if not ast_path.exists() or not cfg_path.exists():
            print(f"[SKIP] {chain}: missing ast or cfg")
            continue

        print(f"\n[BUILD] {chain}")

        cfg_map = load_cfg(chain)
        print(f"  CFG graphs: {len(cfg_map)}")

        out_path = OUT_DIR / f"{chain}.jsonl"
        kept = total = 0

        with ast_path.open("r", encoding="utf8") as fin, \
             out_path.open("w", encoding="utf8") as fout:

            for line in tqdm(fin, desc=f"AST {chain}"):
                try:
                    ast = json.loads(line)
                except Exception:
                    continue

                total += 1
                src_path = ast.get("id")  # AST-LLM 的 id 就是 src_path
                if not src_path:
                    continue

                cfg = cfg_map.get(src_path)
                if cfg is None:
                    continue

                gid = make_graph_id(chain, src_path)

                record = {
                    "id": gid,
                    "chain": chain,
                    "src_path": src_path,
                    "label": 0,
                    "graphs": {
                        "ast": {
                            "id":src_path,
                            "ast_nodes": ast.get("ast_nodes", [])
                        },
                        "cfg": cfg
                    }
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

        print(f"  Kept {kept} / {total} AST graphs")
        print(f"  Output → {out_path}")


if __name__ == "__main__":
    main()
