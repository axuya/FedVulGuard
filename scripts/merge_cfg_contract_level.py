import json
from pathlib import Path
from tqdm import tqdm

IN_DIR = Path("data/graphs_cfg")               # 原函数级 CFG
OUT_DIR = Path("data/graphs_cfg_contract")     # 输出合约级 CFG
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = ["BSC", "Ethereum", "Polygon", "Avalanche", "Fantom", "Arbitrum"]


def merge_chain_cfg(chain: str):
    src_path = IN_DIR / f"{chain}.jsonl"
    if not src_path.exists():
        print(f"[WARN] CFG file not found for chain {chain}: {src_path}")
        return

    print(f"\n[MERGE] Chain = {chain}")
    # id -> aggregator
    agg = {}

    with src_path.open("r", encoding="utf8") as fin:
        for line in tqdm(fin, desc=f"Reading CFG {chain}"):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            cid = item.get("id")
            if cid is None:
                continue

            graph = item.get("graph") or item.get("cfg")
            if not isinstance(graph, dict):
                continue

            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])

            if cid not in agg:
                agg[cid] = {
                    "nodes": [],
                    "edges": [],
                    "next_id": 0,   # 合约级重新编号
                }

            ctx = agg[cid]
            id_map = {}  # 本函数内 old_id -> new_id

            # 1) 合并节点，重新编号
            for n in nodes:
                old_id = n.get("id")
                if old_id is None:
                    continue

                new_id = ctx["next_id"]
                ctx["next_id"] += 1
                id_map[old_id] = new_id

                new_node = {
                    "id": new_id,
                    "type": n.get("type"),
                    "expression": n.get("expression"),
                    # 保留一下 contract/function 信息，方便调试
                    "contract": graph.get("contract"),
                    "function": graph.get("function"),
                    "source_mapping": n.get("source_mapping"),
                }
                ctx["nodes"].append(new_node)

            # 2) 合并边，做 id 映射
            for e in edges:
                # 适配几种常见字段名：src/dst 或 source/dest
                src = e.get("src")
                dst = e.get("dst")
                if src is None or dst is None:
                    src = e.get("source")
                    dst = e.get("dest")

                if src is None or dst is None:
                    continue

                if src not in id_map or dst not in id_map:
                    continue

                ctx["edges"].append({
                    "src": id_map[src],
                    "dst": id_map[dst],
                    "type": e.get("type"),
                })

    # 写出新的合约级 CFG 文件
    out_path = OUT_DIR / f"{chain}.jsonl"
    with out_path.open("w", encoding="utf8") as fout:
        for cid, ctx in tqdm(agg.items(), desc=f"Writing merged CFG {chain}"):
            item = {
                "id": cid,        # 仍然用原来的 file stem，对齐 AST
                "chain": chain,
                "cfg_nodes": ctx["nodes"],
                "cfg_edges": ctx["edges"],
            }
            fout.write(json.dumps(item) + "\n")

    print(f"[OK] Merged CFG saved → {out_path}, total contracts: {len(agg)}")


def main():
    for chain in CHAINS:
        merge_chain_cfg(chain)


if __name__ == "__main__":
    main()
