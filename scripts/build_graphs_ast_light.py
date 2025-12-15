import json
from pathlib import Path
from tqdm import tqdm
from tree_sitter import Language, Parser

# ======================================================
# Load tree-sitter Solidity parser
# ======================================================
SOLIDITY_SO = "build/solidity-languages.so"  # adjust if needed
LANGUAGE = Language(SOLIDITY_SO, "solidity")
print("Loading parser:", SOLIDITY_SO)


parser = Parser()
parser.set_language(LANGUAGE)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/graphs_ast_light")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = ["BSC", "Ethereum", "Polygon", "Avalanche", "Fantom", "Arbitrum"]


def extract_ast_light(node, nodes, edges, parent_id=None, next_id=[0]):
    """
    DFS traversal to extract lightweight AST.
    - nodes[] append { id, type }
    - edges[] append { src, dst, type="AST" }
    """

    nid = next_id[0]
    next_id[0] += 1

    nodes.append({
        "id": nid,
        "type": node.type
    })

    if parent_id is not None:
        edges.append({
            "src": parent_id,
            "dst": nid,
            "type": "AST"
        })

    # Recursively process children
    for child in node.children:
        extract_ast_light(child, nodes, edges, nid, next_id)


def build_light_ast(file_path: str):
    """Parse a .sol file and return lightweight AST graph."""
    try:
        code = Path(file_path).read_text(encoding="utf8", errors="replace")
    except:
        return None

    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception:
        return None

    root = tree.root_node
    nodes = []
    edges = []
    extract_ast_light(root, nodes, edges)

    if len(nodes) == 0:
        return None

    return {
        "nodes": nodes,
        "edges": edges
    }


# ======================================================
# Main runner
# ======================================================
for chain in CHAINS:
    chain_dir = RAW_DIR / chain
    out_file = OUT_DIR / f"{chain}.jsonl"

    with out_file.open("w", encoding="utf8") as fout:
        for sol_file in tqdm(chain_dir.glob("*.sol"), desc=f"Building Light AST {chain}"):

            ast_graph = build_light_ast(str(sol_file))
            if ast_graph is None:
                continue

            item = {
                "id": sol_file.stem,
                "chain": chain,
                "graph": ast_graph
            }
            fout.write(json.dumps(item) + "\n")

    print(f"[OK] Light AST built → {out_file}")
