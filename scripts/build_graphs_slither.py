import os
import json
from pathlib import Path
from tqdm import tqdm
from slither.slither import Slither


def extract_ast(file_path):
    try:
        sl = Slither(file_path)
        # 获取所有合约的AST节点
        result = []
        for contract in sl.contracts:
            result.append({
                "name": contract.name,
                "ast": contract.ast
            })
        return result
    except Exception:
        return None


def main():
    ROOT = "data/raw"
    OUT = "data/graphs_raw"
    CHAINS = ["BSC", "Ethereum", "Polygon", "Avalanche", "Fantom", "Arbitrum"]

    os.makedirs(OUT, exist_ok=True)

    for chain in CHAINS:
        print(f"📌 Processing chain: {chain}")
        chain_dir = Path(ROOT) / chain
        out_file = Path(OUT) / f"{chain}.jsonl"

        with out_file.open("w") as fout:
            for sol in tqdm(chain_dir.glob("*.sol")):
                ast_info = extract_ast(str(sol))
                if ast_info:
                    fout.write(json.dumps({
                        "id": str(sol),
                        "chain": chain,
                        "ast": ast_info
                    }) + "\n")

        print(f"✔ Finished: {out_file}")


if __name__ == "__main__":
    main()
