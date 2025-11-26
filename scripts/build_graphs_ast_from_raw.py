import json
from pathlib import Path
from tqdm import tqdm

RAW_ROOT = Path("data/graphs_ast_raw")
OUT_ROOT = Path("data/graphs_ast_graph")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CHAINS = ["Arbitrum", "Avalanche", "Polygon", "Fantom"]  # 你可以改全链

def build_graph_from_raw_ast(ast_root):
    nodes = []
    edges = []

    def dfs(node, parent_id=None):
        node_id = len(nodes)
        nodes.append({
            "id": node_id,
            "type": node.get("type"),
            "start": node.get("startPosition"),
            "end": node.get("endPosition")
        })

        if parent_id is not None:
            edges.append({
                "src": parent_id,
                "dst": node_id,
                "type": "AST_CHILD"
            })

        for child in node.get("children", []):
            dfs(child, node_id)

    dfs(ast_root)
    return nodes, edges


for chain in CHAINS:
    raw_file = RAW_ROOT / f"{chain}.jsonl"
    out_file = OUT_ROOT / f"{chain}.jsonl"

    if not raw_file.exists():
        print(f"[WARN] Raw file missing: {raw_file}")
        continue

    print(f"\n=== Building AST graphs for {chain} ===")

    with raw_file.open("r", encoding="utf8") as fin, \
         out_file.open("w", encoding="utf8") as fout:

        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except:
                continue

            # 原始结构有 "ast"
            if "ast" not in item:
                continue

            ast_root = item["ast"]
            nodes, edges = build_graph_from_raw_ast(ast_root)

            out = {
                "id": item.get("id"),
                "chain": item.get("chain"),
                "nodes": nodes,
                "edges": edges
            }

            fout.write(json.dumps(out) + "\n")

    print(f"[OK] Written: {out_file}")

