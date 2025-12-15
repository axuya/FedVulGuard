import json
from pathlib import Path
from tqdm import tqdm
import linecache

AST_DIR = Path("data/graphs_ast_norm")
CFG_DIR = Path("data/graphs_cfg_norm")
DFG_DIR = Path("data/graphs_dfg_norm")
OUT_DIR = Path("data/graphs_multigraph_stream")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = ["Arbitrum", "Avalanche", "BSC", "Ethereum", "Fantom", "Polygon"]


def build_line_index(path):
    """
    建立行号索引: id -> line_number
    不用 offset，跨平台稳定，不会出错。
    """
    index = {}
    with open(path, "r", encoding="utf8") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except:
                continue
            gid = obj.get("id")
            if gid:
                index[gid] = lineno
    return index


def load_graph_by_lineno(path, lineno, key_nodes, key_edges):
    line = linecache.getline(str(path), lineno)
    if not line:
        return [], []
    try:
        obj = json.loads(line)
        return obj.get(key_nodes, []), obj.get(key_edges, [])
    except:
        return [], []


def process_chain(chain):
    print(f"\n[STREAM MERGE] chain = {chain}")

    ast_path = AST_DIR / f"{chain}.jsonl"
    cfg_path = CFG_DIR / f"{chain}.jsonl"
    dfg_path = DFG_DIR / f"{chain}.jsonl"

    if not ast_path.exists() or not cfg_path.exists() or not dfg_path.exists():
        print("[WARN] missing files, skip.")
        return

    print("[INFO] Building CFG line index …")
    cfg_index = build_line_index(cfg_path)

    print("[INFO] Building DFG line index …")
    dfg_index = build_line_index(dfg_path)

    out_path = OUT_DIR / f"{chain}.jsonl"
    total = 0

    with open(ast_path, "r", encoding="utf8") as fin, \
         open(out_path, "w", encoding="utf8") as fout:

        for line in tqdm(fin, desc=f"Merging {chain}"):
            try:
                ast_obj = json.loads(line)
            except:
                continue

            gid = ast_obj.get("id")
            if gid not in cfg_index or gid not in dfg_index:
                continue

            cfg_nodes, cfg_edges = load_graph_by_lineno(
                cfg_path, cfg_index[gid], "cfg_nodes", "cfg_edges"
            )
            dfg_nodes, dfg_edges = load_graph_by_lineno(
                dfg_path, dfg_index[gid], "dfg_nodes", "dfg_edges"
            )

            merged = {
                "id": gid,
                "chain": chain,
                "ast_nodes": ast_obj.get("ast_nodes", []),
                "ast_edges": ast_obj.get("ast_edges", []),
                "cfg_nodes": cfg_nodes,
                "cfg_edges": cfg_edges,
                "dfg_nodes": dfg_nodes,
                "dfg_edges": dfg_edges,
            }

            fout.write(json.dumps(merged) + "\n")
            total += 1

    print(f"[OK] {chain}: merged {total} samples → {out_path}")


def main():
    print("=== Multi-graph Streaming Merge (Line-based) ===")
    for chain in CHAINS:
        process_chain(chain)


if __name__ == "__main__":
    main()
