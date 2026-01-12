import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ====== INPUT / OUTPUT ======
JSONL_PATH = Path("data/train/crossgraphnet_lite_labeled/Fantom_500.jsonl")
OUT_DIR = Path("data/embeddings/Fantom_500")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = OUT_DIR / "emb.npy"

# ====== LOCAL OFFLINE CODEBERT ======
MODEL_PATH = "/home/xu/FedVulGuard/CodeBert"  # 你本地离线 CodeBERT 路径

# ====== HYPERPARAMS ======
MAX_LEN = 256
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_code(obj: dict) -> str:
    # Fantom_500.jsonl: expected to have src_path
    sp = obj.get("src_path", "")
    if not isinstance(sp, str) or not sp.strip():
        raise KeyError("Missing src_path in json object")

    fp = Path(sp)
    if fp.exists():
        return fp.read_text(encoding="utf-8", errors="ignore")

    # If src_path is relative, try from repo root
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

    # Save ids.txt (must align with emb.npy order)
    (OUT_DIR / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(codes), BATCH), desc="Embedding"):
            batch_codes = codes[i:i + BATCH]
            enc = tokenizer(
                batch_codes,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            out = model(**enc)  # last_hidden_state: [B, T, H]
            last = out.last_hidden_state

            # mean pooling by attention mask
            mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom  # [B, H], H=768

            embs.append(pooled.detach().cpu().numpy())

    emb = np.concatenate(embs, axis=0)
    assert emb.shape[0] == 500 and emb.shape[1] == 768, emb.shape

    np.save(OUT_NPY, emb.astype(np.float32))
    print("Saved:", OUT_NPY, "shape=", emb.shape)


if __name__ == "__main__":
    main()
