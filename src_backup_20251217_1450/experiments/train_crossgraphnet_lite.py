from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.cuda.amp import autocast, GradScaler

from torch_geometric.data import Data, Batch

from src.crossgraphnet.models.crossgraphnet_lite import CrossGraphNetLite

# -------------------------
# Utility: metrics
# -------------------------
@torch.no_grad()
def compute_f1(logits: torch.Tensor, y: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    eps = 1e-9
    if num_classes == 2:
        tp = ((pred == 1) & (y == 1)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = (pred == y).float().mean().item()
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}
    else:
        # macro-f1 for multi-class (simple implementation)
        f1s = []
        for c in range(num_classes):
            tp = ((pred == c) & (y == c)).sum().item()
            fp = ((pred == c) & (y != c)).sum().item()
            fn = ((pred != c) & (y == c)).sum().item()
            p = tp / (tp + fp + eps)
            r = tp / (tp + fn + eps)
            f1s.append(2 * p * r / (p + r + eps))
        acc = (pred == y).float().mean().item()
        return {"acc": acc, "macro_f1": float(sum(f1s) / len(f1s))}


# -------------------------
# Vocab
# -------------------------
class TypeVocab:
    def __init__(self):
        self.stoi: Dict[str, int] = {"<UNK>": 0}
        self.itos: List[str] = ["<UNK>"]

    def add(self, t: str) -> None:
        if t not in self.stoi:
            self.stoi[t] = len(self.itos)
            self.itos.append(t)

    def encode(self, t: str) -> int:
        return self.stoi.get(t, 0)

    def __len__(self) -> int:
        return len(self.itos)


# -------------------------
# Dataset: expects jsonl lines with either:
# A) {"label":..., "graphs":{"ast":{...}, "cfg":{...}}}
# or B) flat keys like {"ast_nodes":..., "cfg_nodes":..., "cfg_edges":...}
# -------------------------
def _pick_ast_obj(obj: dict) -> dict:
    if "graphs" in obj and isinstance(obj["graphs"], dict):
        return obj["graphs"].get("ast") or {}
    return obj

def _pick_cfg_obj(obj: dict) -> dict:
    if "graphs" in obj and isinstance(obj["graphs"], dict):
        return obj["graphs"].get("cfg") or {}
    return obj

class MultiGraphJsonlDataset(Dataset):
    def __init__(self, paths: List[Path], chains_keep: Optional[set] = None):
        self.items: List[dict] = []
        for p in paths:
            with p.open("r", encoding="utf8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    chain = obj.get("chain")
                    if chains_keep is not None and chain not in chains_keep:
                        continue
                    # label must exist
                    if "label" not in obj:
                        continue
                    self.items.append(obj)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def build_vocabs(dataset: MultiGraphJsonlDataset) -> Tuple[TypeVocab, TypeVocab]:
    ast_vocab, cfg_vocab = TypeVocab(), TypeVocab()
    for obj in dataset.items:
        ast = _pick_ast_obj(obj)
        cfg = _pick_cfg_obj(obj)

        ast_nodes = ast.get("ast_nodes") or ast.get("nodes") or []
        for n in ast_nodes:
            t = n.get("type")
            if t:
                ast_vocab.add(str(t))

        cfg_nodes = cfg.get("cfg_nodes") or cfg.get("nodes") or []
        for n in cfg_nodes:
            t = n.get("type")
            if t:
                cfg_vocab.add(str(t))
    return ast_vocab, cfg_vocab


def to_pyg_graph_ast(ast: dict, ast_vocab: TypeVocab) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    nodes = ast.get("ast_nodes") or ast.get("nodes") or []
    x = torch.tensor([ast_vocab.encode(str(n.get("type", "<UNK>"))) for n in nodes], dtype=torch.long)

    # optional edges (many of your AST variants do not have edges)
    edges = ast.get("ast_edges") or ast.get("edges")
    if edges:
        # accept either [{"src":..,"dst":..}, ...] or [[src,dst],...]
        src_list, dst_list = [], []
        for e in edges:
            if isinstance(e, dict):
                src_list.append(int(e.get("src")))
                dst_list.append(int(e.get("dst")))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                src_list.append(int(e[0]))
                dst_list.append(int(e[1]))
        if len(src_list) > 0:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            return x, edge_index
    return x, None


def to_pyg_graph_cfg(cfg: dict, cfg_vocab: TypeVocab) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes = cfg.get("cfg_nodes") or cfg.get("nodes") or []
    x = torch.tensor([cfg_vocab.encode(str(n.get("type", "<UNK>"))) for n in nodes], dtype=torch.long)

    edges = cfg.get("cfg_edges") or cfg.get("edges") or []
    src_list, dst_list = [], []
    for e in edges:
        if isinstance(e, dict):
            src_list.append(int(e.get("src")))
            dst_list.append(int(e.get("dst")))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            src_list.append(int(e[0]))
            dst_list.append(int(e[1]))
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) if len(src_list) > 0 else torch.empty((2, 0), dtype=torch.long)
    return x, edge_index


def collate_fn(batch: List[dict], ast_vocab: TypeVocab, cfg_vocab: TypeVocab):
    ast_graphs, cfg_graphs, ys, chains = [], [], [], []
    for obj in batch:
        y = int(obj["label"])
        chain = obj.get("chain", "Unknown")

        ast = _pick_ast_obj(obj)
        cfg = _pick_cfg_obj(obj)

        ast_x, ast_edge = to_pyg_graph_ast(ast, ast_vocab)
        cfg_x, cfg_edge = to_pyg_graph_cfg(cfg, cfg_vocab)

        # handle empty graphs
        if ast_x.numel() == 0:
            ast_x = torch.zeros((1,), dtype=torch.long)  # one UNK node
        if cfg_x.numel() == 0:
            cfg_x = torch.zeros((1,), dtype=torch.long)
            cfg_edge = torch.empty((2, 0), dtype=torch.long)

        ast_data = Data(node_type=ast_x, edge_index=ast_edge if ast_edge is not None else torch.empty((2, 0), dtype=torch.long))
        cfg_data = Data(node_type=cfg_x, edge_index=cfg_edge)

        ast_graphs.append(ast_data)
        cfg_graphs.append(cfg_data)
        ys.append(y)
        chains.append(chain)

    ast_batch = Batch.from_data_list(ast_graphs)
    cfg_batch = Batch.from_data_list(cfg_graphs)
    y = torch.tensor(ys, dtype=torch.long)
    return ast_batch, cfg_batch, y, chains


# -------------------------
# Training
# -------------------------
@dataclass
class TrainConfig:
    device: str = "cuda"
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    emb_dim: int = 128
    hidden_dim: int = 128
    gnn_layers: int = 2
    num_classes: int = 2
    use_gated_fusion: bool = False
    patience: int = 5


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Dict[str, float]:
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
    return compute_f1(logits, y, num_classes=num_classes)


def main():
    # -------------------------
    # EDIT HERE: point to your jsonl(s)
    # Recommended: use your aligned multigraph output jsonl(s)
    # -------------------------
    DATA_PATHS = [
        Path("data/train/crossgraphnet_lite_labeled/Ethereum.jsonl"),
        #Path("data/train/crossgraphnet_lite_labeled/BSC.jsonl"),
        # Path("data/graphs_multigraph_uid/BSC.jsonl"),
        # Path("data/graphs_multigraph_uid/Polygon.jsonl"),
    ]

    cfg = TrainConfig()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Example split:
    # Train/Val on Ethereum, Test on other chains: build separate datasets/loaders
    train_ds = MultiGraphJsonlDataset(DATA_PATHS, chains_keep=None)
    
    train_ds.items = train_ds.items[:1000]#限制样本数为1000，用于pipeline的验证
    
    # simple random split
    n = len(train_ds)
    if n < 100:
        raise RuntimeError(f"Too few training samples: {n}. Check labels/data paths.")

    # shuffle indices
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]

    train_subset = MultiGraphJsonlDataset([], None)
    val_subset = MultiGraphJsonlDataset([], None)
    train_subset.items = [train_ds.items[i] for i in train_idx]
    val_subset.items = [train_ds.items[i] for i in val_idx]

    ast_vocab, cfg_vocab = build_vocabs(train_subset)

    # class weights (binary)
    y_train = torch.tensor([int(x["label"]) for x in train_subset.items], dtype=torch.long)
    if cfg.num_classes == 2:
        pos = (y_train == 1).sum().item()
        neg = (y_train == 0).sum().item()
        # avoid div-by-zero
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    model_cfg = CrossGraphNetLiteConfig(
    num_node_types_ast=len(ast_vocab),
    num_node_types_cfg=len(cfg_vocab),
    emb_dim=cfg.emb_dim,
    hidden_dim=cfg.hidden_dim,
    num_layers=cfg.gnn_layers,
    dropout=cfg.dropout,
    num_classes=cfg.num_classes,
    )

    model = CrossGraphNetLite(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    def _make_loader(ds: MultiGraphJsonlDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: collate_fn(b, ast_vocab, cfg_vocab),
        )

    train_loader = _make_loader(train_subset, shuffle=True)
    val_loader = _make_loader(val_subset, shuffle=False)

    best_val = -1.0
    bad = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for ast_b, cfg_b, y, _chains in train_loader:
            ast_b = ast_b.to(device)
            cfg_b = cfg_b.to(device)
            y = y.to(device)

            ast_edge = ast_b.edge_index
            if ast_edge is not None and ast_edge.numel() == 0:
                ast_edge = None

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(
                    ast_type_idx=ast_b.node_type,
                    ast_batch=ast_b.batch,
                    cfg_type_idx=cfg_b.node_type,
                    cfg_edge_index=cfg_b.edge_index,
                    cfg_batch=cfg_b.batch,
                    ast_edge_index=ast_edge,
                )
                if cfg.num_classes == 2:
                    # BCE expects float targets shape [B]
                    loss = criterion(logits[:, 1], y.float())
                else:
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        val_metrics = run_eval(model, val_loader, str(device), cfg.num_classes)
        val_score = val_metrics.get("f1", val_metrics.get("macro_f1", 0.0))

        print(f"Epoch {epoch:03d} | loss={total_loss/len(train_loader):.4f} | val={val_metrics}")

        if val_score > best_val:
            best_val = val_score
            bad = 0
            os.makedirs("models", exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "ast_vocab": ast_vocab.stoi,
                    "cfg_vocab": cfg_vocab.stoi,
                    "config": cfg.__dict__,
                },
                "models/crossgraphnet_lite.pt",
            )
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stop at epoch {epoch}. Best val={best_val:.4f}")
                break

    print("Saved best checkpoint to models/crossgraphnet_lite.pt")


if __name__ == "__main__":
    main()
