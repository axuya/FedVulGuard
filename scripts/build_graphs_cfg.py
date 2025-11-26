import json
from pathlib import Path
from tqdm import tqdm
from slither import Slither

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/graphs_cfg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = ["Arbitrum"]


def safe_str(obj):
    """Convert any Slither object into JSON-safe string."""
    try:
        return str(obj)
    except:
        try:
            return repr(obj)
        except:
            return "<unknown>"


def safe_filename(f):
    """Convert Filename object to string path."""
    try:
        if hasattr(f,"used"):
            return f.used
        if hasattr(f,"ralative"):
            return f.ralative
        return str(f)
    except:
        return "<unknown-file>"


def build_cfg_for_file(sol_path: str):
    try:
        sl = Slither(sol_path)
    except Exception:
        return []

    graphs = []

    for c in sl.contracts:
        for f in c.functions_declared:
            nodes = []
            edges = []

            # build nodes
            for bb in f.nodes:
                node_id = len(nodes)

                # Safe source mapping
                sm = bb.source_mapping
                source_mapping = {
                    "filename": safe_filename(sm.filename),
                    "lines": sm.lines if isinstance(sm.lines, list) else []
                }

                nodes.append({
                    "id": node_id,
                    "type": safe_str(bb.type),
                    "expression": safe_str(bb.expression),
                    "source_mapping": source_mapping
                })

                # tag temporary ID
                bb._tmp_id = node_id

            # build edges
            for bb in f.nodes:
                for succ in bb.sons:
                    edges.append({
                        "src": bb._tmp_id,
                        "dst": succ._tmp_id,
                        "type": "CFG_EDGE"
                    })

            graphs.append({
                "contract": c.name,
                "function": f.full_name,
                "nodes": nodes,
                "edges": edges
            })

    return graphs


for chain in CHAINS:
    out_file = OUT_DIR / f"{chain}.jsonl"
    chain_dir = RAW_DIR / chain

    with out_file.open("w", encoding="utf8") as fout:

        # iterate .sol files
        for sol_file in tqdm(chain_dir.glob("*.sol")):
            graphs = build_cfg_for_file(str(sol_file))
            if not graphs:
                continue

            for g in graphs:
                item = {
                    "id": str(sol_file),
                    "chain": chain,
                    "cfg": g
                }
                fout.write(json.dumps(item) + "\n")

    print(f"[OK] CFG Built: {out_file}")
