import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

JSONL_PATH = Path("data/train/crossgraphnet_lite_labeled/Polygon_500.jsonl")
OUT_DIR = Path("data/embeddings/Polygon_500")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = OUT_DIR / "emb.npy"

# 你离线模型的路径：请改成你本地真实路径（你之前手动下载过）
# 常见：models/codebert-base 或 microsoft/codebert-base
MODEL_PATH = "/home/xu/FedVulGuard/CodeBert"

MAX_LEN = 256
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from pathlib import Path

def get_code(obj: dict) -> str:
    # Polygon_500.jsonl: only has src_path
    sp = obj.get("src_path", "")
    if not isinstance(sp, str) or not sp.strip():
        raise KeyError("Missing src_path in json object")

    fp = Path(sp)
    if fp.exists():
        return fp.read_text(encoding="utf-8", errors="ignore")

    # 兼容：如果存的是相对路径，尝试以项目根目录拼接
    fp2 = Path(".") / sp
    if fp2.exists():
        return fp2.read_text(encoding="utf-8", errors="ignore")

    raise FileNotFoundError(f"src_path not found on disk: {sp}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
    model.eval()

    ids, codes = [], []
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            codes.append(get_code(obj))

    # 保存 ids.txt（确保与 emb.npy 顺序一致）
    (OUT_DIR / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(codes), BATCH), desc="Embedding"):
            batch_codes = codes[i:i+BATCH]
            enc = tokenizer(
                batch_codes,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)

            out = model(**enc)  # last_hidden_state: [B, T, H]
            last = out.last_hidden_state

            # mean pooling（按 attention mask）
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom  # [B, H]，H=768

            embs.append(pooled.detach().cpu().numpy())

    emb = np.concatenate(embs, axis=0)
    assert emb.shape[0] == 500 and emb.shape[1] == 768, emb.shape
    np.save(OUT_NPY, emb)
    print("Saved:", OUT_NPY, "shape=", emb.shape)

if __name__ == "__main__":
    main()
