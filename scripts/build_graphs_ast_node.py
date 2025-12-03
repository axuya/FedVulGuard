import json, subprocess, os
from pathlib import Path
from tqdm import tqdm

ROOT = "data/raw"
OUT = "data/graphs_ast"
os.makedirs(OUT, exist_ok=True)

#CHAINS = ["BSC","Ethereum"]
CHAINS = ["BSC", "Ethereum", "Polygon", "Avalanche", "Fantom", "Arbitrum"]

JS_AST = "scripts/parse_ast_node.js"

def parse_ast(path):
    try:
        res = subprocess.run(
            ["node", JS_AST, str(path)],
            capture_output=True,
            text=True,
            timeout=15
        )
        if res.returncode != 0:
            return None
        return json.loads(res.stdout)
    except:
        return None

for chain in CHAINS:
    chain_dir = Path(ROOT) / chain
    out_file = Path(OUT) / f"{chain}.jsonl"

    print(f"\n=== Processing chain: {chain} ===")

    with out_file.open("w") as fout:
        for sol in tqdm(list(chain_dir.glob("*.sol"))):
            ast = parse_ast(sol)
            if ast is None:
                continue
            fout.write(json.dumps({
                "id": str(sol),
                "chain": chain,
                "ast": ast
            }) + "\n")

    print(f"✔ Finished: {out_file}")
