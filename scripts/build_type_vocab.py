import json
from pathlib import Path

DATA_PATH = Path("data/train/crossgraphnet_lite/Ethereum.jsonl")
OUT_PATH = Path("data/meta/type_vocab_lite.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ast_types = set()
cfg_types = set()

with DATA_PATH.open() as f:
    for line in f:
        obj = json.loads(line)
        for n in obj["graphs"]["ast"]["ast_nodes"]:
            ast_types.add(n["type"])
        for n in obj["graphs"]["cfg"]["cfg_nodes"]:
            cfg_types.add(n["type"])

meta = {
    "num_ast_types": len(ast_types),
    "num_cfg_types": len(cfg_types),
    "ast_types": sorted(ast_types),
    "cfg_types": sorted(cfg_types),
}

with OUT_PATH.open("w") as f:
    json.dump(meta, f, indent=2)

print(f"[OK] Saved vocab meta to {OUT_PATH}")
print(f"AST types: {len(ast_types)}")
print(f"CFG types: {len(cfg_types)}")
