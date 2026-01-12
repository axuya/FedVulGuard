import json, random
from pathlib import Path
from collections import Counter, defaultdict

IN_PATH  = Path("data/train/crossgraphnet_lite_labeled/Polygon.jsonl")
OUT_PATH = Path("data/train/crossgraphnet_lite_labeled/Polygon_500.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
N_TOTAL = 500
N_POS = 250
N_NEG = 250

random.seed(SEED)

pos, neg, other = [], [], []
with IN_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        y = obj.get("label", None)
        if y == 1:
            pos.append(obj)
        elif y == 0:
            neg.append(obj)
        else:
            other.append(obj)

# 分层抽样
take_pos = min(N_POS, len(pos))
take_neg = min(N_NEG, len(neg))

sample = random.sample(pos, take_pos) + random.sample(neg, take_neg)

# 不足则从剩余补齐（保持可复现）
remain = N_TOTAL - len(sample)
if remain > 0:
    pool = pos[take_pos:] + neg[take_neg:] + other
    if len(pool) < remain:
        raise RuntimeError(f"Not enough samples: need {remain}, but pool has {len(pool)}")
    sample += random.sample(pool, remain)

random.shuffle(sample)

with OUT_PATH.open("w", encoding="utf-8") as f:
    for obj in sample:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

dist = Counter([o.get("label", None) for o in sample])
print("Wrote:", OUT_PATH)
print("Total:", len(sample), "LabelDist:", dict(dist))
