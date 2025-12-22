import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SRC = Path("data/train/crossgraphnet_lite_labeled/Ethereum_500.jsonl")
OUT = Path("data/train/crossgraphnet_lite_labeled/Ethereum_500_llm.jsonl")


def normalize_code(code: str) -> str:
    if not code:
        return "emptycode"

    # 去多行注释 /* */
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    # 去单行注释 //
    code = re.sub(r"//.*", " ", code)
    # 把符号换成空格
    code = re.sub(r"[^A-Za-z0-9_]", " ", code)
    # 压缩空格
    code = re.sub(r"\s+", " ", code).strip()

    return code if code else "emptycode"


rows = []
texts = []

with SRC.open("r", encoding="utf8") as f:
    for line in f:
        obj = json.loads(line)

        # 尝试多种源码字段
        code = (
            obj.get("src_code")
            or obj.get("code")
            or obj.get("source")
            or ""
        )

        code = normalize_code(code)

        rows.append(obj)
        texts.append(code)

print("Example normalized text:")
print(texts[0][:200])

vectorizer = TfidfVectorizer(
    max_features=768,
    lowercase=True,
    min_df=1,
)

X = vectorizer.fit_transform(texts)
X = X.toarray().astype(np.float32)

with OUT.open("w", encoding="utf8") as f:
    for obj, emb in zip(rows, X):
        obj["llm_emb"] = emb.tolist()
        f.write(json.dumps(obj) + "\n")

print("Saved:", OUT)
print("Embedding shape:", X.shape)
