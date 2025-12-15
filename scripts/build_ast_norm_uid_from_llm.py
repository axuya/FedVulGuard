import json
import hashlib
from pathlib import Path

def make_graph_id(chain, src_path):
    raw = f"{chain}::{src_path}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def main():
    ast_dir = Path("data/graphs_ast_llm")
    cfg_index_dir = Path("data/index")   # 你之前生成的
    out_dir = Path("data/graphs_ast_norm_uid")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ast_file in ast_dir.glob("*.jsonl"):
        chain = ast_file.stem
        index_path = cfg_index_dir / f"{chain}_src2id.json"
        if not index_path.exists():
            print(f"[SKIP] {chain}: no cfg index")
            continue

        src2id = json.loads(index_path.read_text(encoding="utf8"))
        out_path = out_dir / ast_file.name

        total = 0
        kept = 0

        with ast_file.open("r", encoding="utf8") as fin, \
             out_path.open("w", encoding="utf8") as fout:

            for line in fin:
                g = json.loads(line)
                total += 1

                # 关键：LLM AST 的 id 本身就是 src_path
                src_path = g.get("id")
                if not src_path:
                    continue

                # 只保留 CFG 中也存在的合约（保证可多图融合）
                if src_path not in src2id:
                    continue

                g["src_path"] = src_path
                g["chain"] = chain
                g["id"] = src2id[src_path]  # 统一 graph_id

                fout.write(json.dumps(g, ensure_ascii=False) + "\n")
                kept += 1

        print(f"[AST UID] {chain}: kept {kept}/{total}")

if __name__ == "__main__":
    main()
