import json
import hashlib
from pathlib import Path
from tqdm import tqdm

def hash_id(path_str):
    return hashlib.sha1(path_str.encode()).hexdigest()

def normalize_file(in_path, out_path, key="id"):
    with open(in_path, "r", encoding="utf8") as fin, open(out_path, "w", encoding="utf8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except:
                continue

            # 统一 ID
            if "src_path" in obj:
                new_id = hash_id(obj["src_path"])
            elif key in obj:
                new_id = hash_id(obj[key])
            else:
                continue

            obj["id"] = new_id
            fout.write(json.dumps(obj) + "\n")

def process_dir(in_dir, out_dir, key="id"):
    out_dir.mkdir(exist_ok=True, parents=True)
    for f in in_dir.glob("*.jsonl"):
        normalize_file(f, out_dir / f.name, key)

if __name__ == "__main__":
    base = Path("data")

    # AST
    process_dir(base/"graphs_ast_llm_fixed", base/"graphs_ast_norm")

    # CFG
    process_dir(base/"graphs_cfg_contract", base/"graphs_cfg_norm")

    # DFG
    process_dir(base/"graphs_dfg", base/"graphs_dfg_norm")
