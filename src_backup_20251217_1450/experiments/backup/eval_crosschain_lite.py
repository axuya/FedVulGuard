from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from src.crossgraphnet.models.crossgraphnet_lite import CrossGraphNetLite


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

meta = json.load(open("data/meta/type_vocab_lite.py"))

num_ast_types = meta["num_ast_types"]
num_cfg_types = meta["num_cfg_types"]

# =========================
# Dataset（与 train 脚本同构）
# =========================
class CrossGraphJsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.items = []
        with path.open("r", encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        obj = self.items[idx]

        ast_nodes = obj["graphs"]["ast"]["ast_nodes"]
        cfg_nodes = obj["graphs"]["cfg"]["cfg_nodes"]
        cfg_edges = obj["graphs"]["cfg"]["cfg_edges"]

        # AST graph（无边）
        ast_x = torch.tensor(
            [[hash(n["type"]) % 10000] for n in ast_nodes],
            dtype=torch.long,
        )
        ast_data = Data(x=ast_x)

        # CFG graph
        cfg_x = torch.tensor(
            [[hash(n["type"]) % 10000] for n in cfg_nodes],
            dtype=torch.long,
        )
        if cfg_edges:
            edge_index = torch.tensor(
                [[e["src"], e["dst"]] for e in cfg_edges],
                dtype=torch.long,
            ).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        cfg_data = Data(x=cfg_x, edge_index=edge_index)

        y = torch.tensor(obj["label"], dtype=torch.long)

        return {
            "ast": ast_data,
            "cfg": cfg_data,
            "y": y,
        }


def collate_fn(batch: List[dict]):
    ast_list = [b["ast"] for b in batch]
    cfg_list = [b["cfg"] for b in batch]
    y = torch.stack([b["y"] for b in batch])

    return {
        "ast": Batch.from_data_list(ast_list),
        "cfg": Batch.from_data_list(cfg_list),
        "y": y,
    }


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        ast = batch["ast"].to(DEVICE)
        cfg = batch["cfg"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        logits = model(ast, cfg)
        preds = (torch.sigmoid(logits) > 0.5).long()

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =========================
# Main
# =========================
def main():
    ckpt = Path("models/crossgraphnet_lite.pt")
    print(f"[INFO] Loading model: {ckpt}")

    #model = CrossGraphNetLite.load_from_checkpoint(ckpt)
    model = CrossGraphNetLite(
    num_ast_types=num_ast_types,
    num_cfg_types=num_cfg_types,
    hidden_dim=128,
    num_layers=3,
    dropout=0.1,
    )

    state = torch.load(ckpt,map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)

    datasets = {
        "BSC": Path("data/train/crossgraphnet_lite_labeled/BSC.jsonl"),
        "Fantom": Path("data/train/crossgraphnet_lite_labeled/Fantom.jsonl"),
        "Avalanche": Path("data/train/crossgraphnet_lite_labeled/Avalanche.jsonl"),
    }

    print("\n=== Cross-chain Evaluation (Train: Ethereum) ===")

    for name, path in datasets.items():
        print(f"\n[Eval] Ethereum → {name}")

        ds = CrossGraphJsonlDataset(path)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        metrics = evaluate(model, loader)

        print(
            f"acc={metrics['acc']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
