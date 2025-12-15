import json
from pathlib import Path

def main():
    ast_dir = Path("data/graphs_ast_norm")
    index_dir = Path("data/index")
    out_dir = Path("data/graphs_ast_norm_uid")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ast_file in ast_dir.glob("*.jsonl"):
        chain = ast_file.stem
        index_path = index_dir / f"{chain}_src2id.json"
        if not index_path.exists():
            continue

        src2id = json.loads(index_path.read_text(encoding="utf8"))
        out_path = out_dir / ast_file.name

        fixed = 0
        total = 0

        with ast_file.open("r", encoding="utf8") as fin, \
             out_path.open("w", encoding="utf8") as fout:

            for line in fin:
                g = json.loads(line)
                total += 1

                # 你这里 AST 没有 src_path，只能跳过
                # 但如果你之前保留过 filename/path，可以在这里改
                src_path = g.get("src_path")
                if not src_path or src_path not in src2id:
                    continue

                g["src_path"] = src_path
                g["id"] = src2id[src_path]
                g["chain"] = chain

                fout.write(json.dumps(g, ensure_ascii=False) + "\n")
                fixed += 1

        print(f"[AST FIX] {chain}: fixed {fixed}/{total}")

if __name__ == "__main__":
    main()
