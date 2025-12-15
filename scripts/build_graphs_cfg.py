import json
from pathlib import Path
from tqdm import tqdm
from slither import Slither

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/graphs_cfg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = ["BSC", "Ethereum", "Polygon", "Avalanche", "Fantom", "Arbitrum"]


def safe_str(obj):
    try:
        return str(obj)
    except:
        try:
            return repr(obj)
        except:
            return "<unknown>"


def safe_filename(f):
    try:
        if hasattr(f, "used"):
            return f.used
        if hasattr(f, "ralative"):
            return f.ralative
        return str(f)
    except:
        return "<unknown-file>"


def build_cfg_for_file(sol_path: str):
    """Build a single unified CFG for the whole file (multi-contract, multi-function)."""

    try:
        sl = Slither(sol_path)
    except Exception:
        return None

    all_nodes = []
    all_edges = []

    # global id counter across all functions
    gid = 0

    # Loop contracts
    for c in sl.contracts:
        # Loop functions
        for f in c.functions_declared:
            local_to_global = {}

            # Build nodes
            for bb in f.nodes:
                global_id = gid
                gid += 1

                sm = bb.source_mapping
                source_mapping = {
                    "filename": safe_filename(sm.filename),
                    "lines": sm.lines if isinstance(sm.lines, list) else []
                }

                all_nodes.append({
                    "id": global_id,
                    "contract": c.name,
                    "function": f.full_name,
                    "type": safe_str(bb.type),
                    "expression": safe_str(bb.expression),
                    "source_mapping": source_mapping
                })

                local_to_global[bb] = global_id

            # Build edges
            for bb in f.nodes:
                for succ in bb.sons:
                    all_edges.append({
                        "src": local_to_global[bb],
                        "dst": local_to_global[succ],
                        "type": "CFG_EDGE"
                    })

    # If empty, return None
    if len(all_nodes) == 0:
        return None

    return {
        "nodes": all_nodes,
        "edges": all_edges
    }


# Main executor
for chain in CHAINS:
    out_file = OUT_DIR / f"{chain}.jsonl"
    chain_dir = RAW_DIR / chain

    with out_file.open("w", encoding="utf8") as fout:
        for sol_file in tqdm(chain_dir.glob("*.sol"), desc=f"Building CFG {chain}"):
            cfg_graph = build_cfg_for_file(str(sol_file))
            if cfg_graph is None:
                continue

            item = {
                "id": sol_file.stem,  # match AST/DFG id
                "chain": chain,
                "graph": cfg_graph
            }
            fout.write(json.dumps(item) + "\n")

    print(f"[OK] CFG Built → {out_file}")
