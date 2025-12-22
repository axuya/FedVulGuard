import json

path = "data/train/crossgraphnet_lite_labeled/Ethereum.jsonl"

with open(path) as f:
    obj = json.loads(next(f))

print(obj["graphs"]["ast"]["nodes"][:5])
print(obj["graphs"]["cfg"]["nodes"][:5])
