import os
import json
from pathlib import Path
from tqdm import tqdm

from tree_sitter import Language,Parser

SOLIDITY = Language("build/solidity.so","solidity")
parser = Parser()
parser.set_language(SOLIDITY)

ROOT = Path("data/raw")
OUT = Path("data/graphs_ast")
OUT.mkdir(parents=True, exist_ok=True)

CHAINS = ["BSC","Ethereum"]
#CHAINS = ["BSC", "Ethereum", "Polygon", "Avalanche", "Fantom", "Arbitrum"]
#重新跑BSC和ETH，Arbitrum都MV到graphs_ast_raw去了
def build_graph(node, node_list, edge_list, parent_id=None):
    node_id = len(node_list)
    node_list.append({
        "id": node_id,
        "type": node.type,
        "start": node.start_byte,
        "end": node.end_byte
    })
    if parent_id is not None:
        edge_list.append({
            "src": parent_id,
            "dst": node_id,
            "type": "AST_CHILD"
        })
    for child in node.children:
        build_graph(child, node_list, edge_list, node_id)

def parse_file(path: Path):
    try:
        code = path.read_text(encoding="utf8", errors="ignore")
        tree = parser.parse(code.encode("utf8"))
        root = tree.root_node
        nodes, edges = [], []
        build_graph(root, nodes, edges)
        return nodes, edges
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return None

for chain in CHAINS:
    print(f"\n=== Processing {chain} ===")
    chain_dir = ROOT / chain
    out_file = OUT / f"{chain}.jsonl"

    sol_files = list(chain_dir.glob("*.sol"))
    print(f"[INFO] {len(sol_files)} files")

    with out_file.open("w", encoding="utf8") as fout:
        for f in tqdm(sol_files):
            result = parse_file(f)
            if result:
                nodes, edges = result
                item = {
                    "id": f.name,
                    "path": str(f),
                    "nodes": nodes,
                    "edges": edges
                }
                fout.write(json.dumps(item) + "\n")

    print(f"[OK] Output: {out_file}")
