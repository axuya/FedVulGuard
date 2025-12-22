#这个代码需要把数据进行整合，先用旧的跑通再说


from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#from data_lite import MultiGraphJsonlDataset, build_vocabs, collate_fn, TypeVocab
from src.experiments.data_lite import (
    MultiGraphJsonlDataset,
    build_vocabs,
    collate_fn,
    TypeVocab,
)

from src.crossgraphnet.models.crossgraphnet_lite import (
    CrossGraphNetLite,
    CrossGraphNetLiteConfig,
)



@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()

    # binary metrics for class=1 as positive
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    for ast_b, cfg_b, y, _chains in loader:
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
        all_logits.append(logits)
        all_y.append(y)

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    return compute_metrics(logits, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True, help="Train JSONL path")
    ap.add_argument("--val", type=str, required=True, help="Val JSONL path")
    ap.add_argument("--out", type=str, required=True, help="Output checkpoint path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train_path = Path(args.train)
    val_path = Path(args.val)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = MultiGraphJsonlDataset([train_path])
    val_ds = MultiGraphJsonlDataset([val_path])

    ast_vocab, cfg_vocab = build_vocabs(train_ds)
    cfg = CrossGraphNetLiteConfig(
        num_node_types_ast=len(ast_vocab),
        num_node_types_cfg=len(cfg_vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossGraphNetLite(cfg).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, ast_vocab, cfg_vocab),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, ast_vocab, cfg_vocab),
    )

    # class weight (optional): compute from train quickly
    pos = 0
    for obj in train_ds.items:
        if int(obj["label"]) == 1:
            pos += 1
    neg = len(train_ds) - pos
    if pos > 0 and neg > 0:
        w0 = (pos + neg) / (2.0 * neg)
        w1 = (pos + neg) / (2.0 * pos)
        class_weight = torch.tensor([w0, w1], dtype=torch.float, device=device)
    else:
        class_weight = None

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for ast_b, cfg_b, y, _chains in train_loader:
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
            loss = criterion(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            n_batches += 1

        val_metrics = run_eval(model, val_loader, device)
        avg_loss = total_loss / max(1, n_batches)

        print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "model_state": best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_cfg": asdict(cfg),
        "ast_vocab": ast_vocab.stoi,   # portable dict
        "cfg_vocab": cfg_vocab.stoi,
        "best_val_f1": best_f1,
    }
    torch.save(ckpt, out_path)
    print(f"Saved checkpoint: {out_path} (best_val_f1={best_f1:.4f})")


if __name__ == "__main__":
    main()
