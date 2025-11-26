import json, subprocess, os
from pathlib import Path
from tqdm import tqdm

ROOT = "data/raw"
OUT = "data/graphs_ast"

os.makedirs(OUT, exist_ok=True)

JS_PARSER = "scripts/parse_ast.js"

CHAINS = ["BSC", "Ethereum", "Polygon", "Fantom", "Avalanche", "Arbitrum"]

for chain in CHAINS:
    print(f"Processing chain: {chain}")
    chain_dir = Path(ROOT) / chain
    out_file = Path(OUT) / f"{chain}.jsonl"

    with out_file.open("w") as fout:
        for sol in tqdm(list(chain_dir.glob("*.sol"))):
            try:
                res = subprocess.run(
                    ["node", JS_PARSER, str(sol)],
                    capture_output=True,
                    timeout=20
                )
                if res.returncode != 0:
                    continue

                ast = json.loads(res.stdout.decode())
                fout.write(json.dumps({
                    "id": str(sol),
                    "chain": chain,
                    "ast": ast
                }) + "\n")
            except:
                pass

    print(f"✔ Finished {chain}: {out_file}")
