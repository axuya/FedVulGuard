import json
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm

SRC = Path("data/train/crossgraphnet_lite_labeled/Ethereum_500.jsonl")
OUT = Path("data/train/crossgraphnet_lite_labeled/Ethereum_500_structsem.jsonl")

MAX_DIM = 128  # 结构语义维度（可写进论文）

rows = []
all_types = Counter()

# 第一次遍历：收集 node type
with SRC.open("r", encoding="utf8") as f:
    for line in f:
        obj = json.loads(line)

        # 尝试所有可能的 AST 布局
        ast = (
            obj.get("graphs", {}).get("ast")
            or obj.get("ast")
            or obj
        )

        nodes = ast.get("nodes", [])
        for n in nodes:
            t = n.get("type")
            if t is not None:
                all_types[str(t)] += 1

        rows.append(obj)

# 选最常见的 MAX_DIM 种 type
top_types = [t for t, _ in all_types.most_common(MAX_DIM)]
type2idx = {t: i for i, t in enumerate(top_types)}

print("Collected AST types:", len(type2idx))

# 第二次遍历：构造 embedding
with OUT.open("w", encoding="utf8") as f:
    for obj in tqdm(rows):
        ast = (
            obj.get("graphs", {}).get("ast")
            or obj.get("ast")
            or obj
        )
        nodes = ast.get("nodes", [])

        vec = np.zeros(MAX_DIM, dtype=np.float32)
        for n in nodes:
            t = n.get("type")
            if t in type2idx:
                vec[type2idx[t]] += 1.0

        # 归一化（可选，但推荐）
        if vec.sum() > 0:
            vec = vec / vec.sum()

        obj["llm_emb"] = vec.tolist()
        f.write(json.dumps(obj) + "\n")

print("Saved:", OUT)

