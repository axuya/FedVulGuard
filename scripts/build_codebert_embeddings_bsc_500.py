import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ===== paths =====
DATA_PATH = Path("data/train/crossgraphnet_lite_labeled/BSC_500.jsonl")
OUT_DIR = Path("data/embeddings/BSC_500")
MODEL_PATH = "/home/xu/FedVulGuard/CodeBert"


OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== load model =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
model.eval()

ids = []
embs = []

with DATA_PATH.open("r", encoding="utf8") as f:
    for i, line in enumerate(tqdm(f, total=500)):
        if i >= 500:
            break

        obj = json.loads(line)

        cid = obj.get("id", f"idx_{i}")
        code = obj.get("source", "")

        if not isinstance(code, str) or len(code.strip()) == 0:
            code = "empty contract"

        inputs = tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0)  # [768]

        ids.append(cid)
        embs.append(emb.cpu().numpy())

# ===== save =====
np.save(OUT_DIR / "emb.npy", np.stack(embs))
with (OUT_DIR / "ids.txt").open("w") as f:
    for cid in ids:
        f.write(cid + "\n")

print(f"Saved {len(ids)} CodeBERT embeddings")

