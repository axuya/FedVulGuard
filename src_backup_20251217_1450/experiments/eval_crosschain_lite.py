from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.experiments.data_lite import (
    MultiGraphJsonlDataset,
    collate_fn,
    TypeVocab,
)

from src.crossgraphnet.models.crossgraphnet_lite import (
    CrossGraphNetLite,
    CrossGraphNetLiteConfig,
)




@torch.no_grad()
def compute_metrics_from_preds(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    acc = (pred == y).float().mean().item()

    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    model.eval()

    all_pred, all_y = [], []
    per_chain_pred = defaultdict(list)
    per_chain_y = defaultdict(list)

    for ast_b, cfg_b, y, chains in loader:
        ast_b = ast_b.to(device)
        cfg_b = cfg_b.to(device)
        y = y.to(device)

        ast_edge = ast_b.edge_index
        if ast_edge is not None and ast_edge.numel() == 0:
            ast_edge = None

        logits = model(
            ast_type_idx=ast_b.node_type,
            ast_batch=ast_b.batch,
            cfg_type_idx=cfg_b.node_type,
            cfg_edge_index=cfg_b.edge_index,
            cfg_batch=cfg_b.batch,
            ast_edge_index=ast_edge,
        )
        pred = logits.argmax(dim=-1)

        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())

        # per-chain: chains length == batch size
        for p_i, y_i, c in zip(pred.detach().cpu().tolist(), y.detach().cpu().tolist(), chains):
            per_chain_pred[c].append(p_i)
            per_chain_y[c].append(y_i)

    all_pred = torch.tensor([v for t in all_pred for v in t.tolist()], dtype=torch.long)
    all_y = torch.tensor([v for t in all_y for v in t.tolist()], dtype=torch.long)

    overall = compute_metrics_from_preds(all_pred, all_y)

    by_chain = {}
    for c in sorted(per_chain_pred.keys()):
        p = torch.tensor(per_chain_pred[c], dtype=torch.long)
        yy = torch.tensor(per_chain_y[c], dtype=torch.long)
        by_chain[c] = compute_metrics_from_preds(p, yy)
        by_chain[c]["n"] = float(len(p))
    overall["n"] = float(len(all_pred))
    return overall, by_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    ap.add_argument("--data", type=str, required=True, help="JSONL evaluation file")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chains", type=str, default="", help="Comma-separated chains to keep (optional)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    ast_vocab = TypeVocab.from_stoi(ckpt["ast_vocab"])
    cfg_vocab = TypeVocab.from_stoi(ckpt["cfg_vocab"])

    cfg_dict = ckpt["model_cfg"]
    cfg = CrossGraphNetLiteConfig(**cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossGraphNetLite(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    keep = None
    if args.chains.strip():
        keep = {c.strip() for c in args.chains.split(",") if c.strip()}

    ds = MultiGraphJsonlDataset([Path(args.data)], chains_keep=keep)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=lambda b: collate_fn(b, ast_vocab, cfg_vocab))

    overall, by_chain = evaluate(model, loader, device)

    print("=== CrossGraphNet-Lite Evaluation ===")
    print(f"Total n={int(overall['n'])} acc={overall['acc']:.4f} f1={overall['f1']:.4f} precision={overall['precision']:.4f} recall={overall['recall']:.4f}")
    print("")
    print("=== Per-chain ===")
    for c, m in by_chain.items():
        print(f"{c:12s} n={int(m['n']):6d} acc={m['acc']:.4f} f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}")


if __name__ == "__main__":
    main()
