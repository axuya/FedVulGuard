import os
import json
from tqdm import tqdm

AST_DIR = "data/graphs_ast"
CFG_DIR = "data/graphs_cfg"
DFG_DIR = "data/graphs_dfg"
OUT_DIR = "data/graphs_raw"


def stream_jsonl(path):
    """流式读取 jsonl"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def merge_records(ast_item, cfg_item, dfg_item):
    """
    合并一条记录成为 MGVD 结构
    """

    # 校验 ID 是否一致（如不一致，至少检查 chain）
    chain = ast_item.get("chain") or cfg_item.get("chain") or dfg_item.get("chain")

    return {
        "id": ast_item.get("id"),
        "chain": chain,

        # AST 整体结构
        "ast": ast_item.get("ast", {}),

        # CFG 边
        "cfg_edges": cfg_item.get("cfg_edges", []),
        "functions": cfg_item.get("functions", []),

        # DFG 边
        "dfg_edges": dfg_item.get("dfg_edges", []),
        "variables": dfg_item.get("variables", []),
    }


def process_chain(chain):
    print(f"[START MGVD MERGE] {chain}")

    ast_path = f"{AST_DIR}/{chain}.jsonl"
    cfg_path = f"{CFG_DIR}/{chain}.jsonl"
    dfg_path = f"{DFG_DIR}/{chain}.jsonl"
    out_path = f"{OUT_DIR}/{chain}.jsonl"

    # 如果某链没有某个文件则跳过
    if not (os.path.exists(ast_path) and os.path.exists(cfg_path) and os.path.exists(dfg_path)):
        print(f"[SKIP] Missing files for {chain}")
        return

    with open(out_path, "w", encoding="utf-8") as fout:
        for ast_item, cfg_item, dfg_item in tqdm(
            zip(stream_jsonl(ast_path), stream_jsonl(cfg_path), stream_jsonl(dfg_path)),
            desc=f"Merging {chain}"
        ):
            merged = merge_records(ast_item, cfg_item, dfg_item)
            fout.write(json.dumps(merged) + "\n")

    print(f"[OK] MGVD Built → {out_path}")


def main():
    ensure_dirs()

    chains = [
        f.split(".")[0]
        for f in os.listdir(AST_DIR)
        if f.endswith(".jsonl")
    ]

    for chain in chains:
        process_chain(chain)


if __name__ == "__main__":
    main()
