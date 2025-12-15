import json
from pathlib import Path
from tqdm import tqdm

AST_DIR = Path("data/graphs_ast_llm")
OUT_DIR = Path("data/graphs_ast_llm_fixed")
OUT_DIR.mkdir(exist_ok=True, parents=True)

CHAINS = ["Arbitrum", "Avalanche", "BSC", "Ethereum", "Fantom", "Polygon"]

def normalize_ast_id(ast_id):
    """
    AST id example:
        data/raw/BSC/145fe16094b558961a527a5e239df7f1f37b488f_CAKEBACK.sol
    Convert to:
        145fe16094b558961a527a5e239df7f1f37b488f_CAKEBACK
    """
    name = Path(ast_id).name.replace(".sol", "")
    return name


def process_chain(chain):
    src = AST_DIR / f"{chain}.jsonl"
    dst = OUT_DIR / f"{chain}.jsonl"

    if not src.exists():
        print(f"[WARN] No AST for {chain}, skip.")
        return

    with src.open("r", encoding="utf8") as fin, \
         dst.open("w", encoding="utf8") as fout:

        for line in tqdm(fin, desc=f"Fix AST {chain}"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except:
                continue

            old_id = item.get("id")
            if old_id:
                item["id"] = normalize_ast_id(old_id)

            fout.write(json.dumps(item) + "\n")

    print(f"[OK] Fixed AST → {dst}")


def main():
    print("=== Fixing AST IDs ===")
    for chain in CHAINS:
        process_chain(chain)

if __name__ == "__main__":
    main()
