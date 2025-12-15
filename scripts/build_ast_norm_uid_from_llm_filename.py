import json
from pathlib import Path

def main():
    ast_dir = Path("data/graphs_ast_llm")
    idx_dir = Path("data/index")
    out_dir = Path("data/graphs_ast_norm_uid_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ast_file in ast_dir.glob("*.jsonl"):
        chain = ast_file.stem
        idx_path = idx_dir / f"{chain}_filename2id.json"
        if not idx_path.exists():
            continue

        filename2id = json.loads(idx_path.read_text(encoding="utf8"))
        out_path = out_dir / ast_file.name

        total = kept = 0

        with ast_file.open("r", encoding="utf8") as fin, \
             out_path.open("w", encoding="utf8") as fout:

            for line in fin:
                g = json.loads(line)
                total += 1

                src_path = g.get("id")
                if not src_path:
                    continue

                filename = Path(src_path).name
                if filename not in filename2id:
                    continue

                g["src_path"] = src_path
                g["chain"] = chain
                g["id"] = filename2id[filename]

                fout.write(json.dumps(g, ensure_ascii=False) + "\n")
                kept += 1

        print(f"[AST UID v2] {chain}: kept {kept}/{total}")

if __name__ == "__main__":
    main()
