import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SRC = Path("data/train/crossgraphnet_lite_labeled/Ethereum_500.jsonl")
OUT = Path("data/train/crossgraphnet_lite_labeled/Ethereum_500_llm.jsonl")

def normalize_text(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    # 去注释
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//.*", " ", s)
    # 保留字母数字下划线
    s = re.sub(r"[^A-Za-z0-9_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ast_types_as_text(obj: dict, max_nodes: int = 3000) -> str:
    """
    Try all possible AST layouts to extract node type tokens.
    This function is intentionally redundant for robustness.
    """
    candidates = []

    # 常见布局 1
    if "graphs" in obj and "ast" in obj["graphs"]:
        candidates.append(obj["graphs"]["ast"])

    # 常见布局 2
    if "ast" in obj:
        candidates.append(obj["ast"])

    # 常见布局 3：直接平铺
    candidates.append(obj)

    toks = []
    for ast in candidates:
        nodes = ast.get("nodes")
        if isinstance(nodes, list) and len(nodes) > 0:
            for n in nodes[:max_nodes]:
                t = n.get("type")
                if t is not None:
                    toks.append(str(t))
            break  # 一旦找到就停

    return " ".join(toks)


rows, texts = [], []

with SRC.open("r", encoding="utf8") as f:
    for line in f:
        obj = json.loads(line)

        code = obj.get("src_code") or obj.get("code") or obj.get("source") or obj.get("src") or obj.get("text") or ""
        code = normalize_text(code)

        if not code:
            code = normalize_text(ast_types_as_text(obj))
        if not code:
            code = "emptycode"  # 最终兜底，确保永不空

        rows.append(obj)
        texts.append(code)

print("Loaded:", len(texts))
print("Example text (first 200 chars):", texts[0][:200])

vectorizer = TfidfVectorizer(
    max_features=768,
    lowercase=True,
    min_df=1,
    token_pattern=r"(?u)\b\w+\b",  # 放宽：1 个字符也算 token，避免极端情况
)

X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
print("Embedding shape:", X.shape)

with OUT.open("w", encoding="utf8") as f:
    for obj, emb in zip(rows, X):
        obj["llm_emb"] = emb.tolist()
        f.write(json.dumps(obj) + "\n")

print("Saved:", OUT)
