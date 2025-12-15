import os
import json
from tqdm import tqdm

AST_DIR = "data/graphs_ast"
CFG_DIR = "data/graphs_cfg"
DFG_DIR = "data/graphs_dfg"
OUT_DIR = "data/graphs_mgvd"

os.makedirs(OUT_DIR, exist_ok=True)


def load_jsonl(path):
    data = {}
    if not os.path.exists(path):
        print(f"[WARN] Not found: {path}")
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["id"]] = item
    return data


def merge_chain(chain):
    print(f"\n[START MERGE] {chain}")

    ast_path = os.path.join(AST_DIR, f"{chain}.jsonl")
    cfg_path = os.path.join(CFG_DIR, f"{chain}.jsonl")
    dfg_path = os.path.join(DFG_DIR, f"{chain}.jsonl")

    ast_data = load_jsonl(ast_path)
    cfg_data = load_jsonl(cfg_path)
    dfg_data = load_jsonl(dfg_path)

    out_path = os.path.join(OUT_DIR, f"{chain}.jsonl")
    fout = open(out_path, "w", encoding="utf-8")

    ids = set(ast_data.keys()) | set(cfg_data.keys()) | set(dfg_data.keys())
    print(f"[INFO] Total unique ids for chain {chain}: {len(ids)}")

    missing_cnt = 0
    matched_cnt = 0

    for cid in tqdm(ids):
        entry = {
            "id": cid,
            "chain": chain,
            "ast": ast_data.get(cid, {}).get("graph"),
            "cfg": cfg_data.get(cid, {}).get("graph"),
            "dfg": dfg_data.get(cid, {}).get("graph"),
            "src_path": ast_data.get(cid, {}).get("src_path")
                       or cfg_data.get(cid, {}).get("src_path")
                       or dfg_data.get(cid, {}).get("src_path"),
        }

        if entry["ast"] and entry["cfg"] and entry["dfg"]:
            matched_cnt += 1
        else:
            missing_cnt += 1

        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    fout.close()

    print(f"[DONE MERGE] {chain} → {out_path}")
    print(f" - Matched (AST+CFG+DFG): {matched_cnt}")
    print(f" - Missing parts:         {missing_cnt}")


def main():
    chains = []
    
    # autodetect chains based on existing AST files
    for name in os.listdir(AST_DIR):
        if name.endswith(".jsonl"):
            chains.append(name.replace(".jsonl", ""))

    print("[INFO] Chains detected:", chains)

    for chain in chains:
        merge_chain(chain)


if __name__ == "__main__":
    main()
