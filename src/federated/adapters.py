# src/federated/adapters.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data_lite import MultiGraphJsonlDataset, build_vocabs, collate_fn
from src.model import CrossGraphNetLite, CrossGraphNetLiteConfig
from src.train_crosschain import evaluate as eval_crosschain


@dataclass
class FLContext:
    chains: List[str]
    data_root: Path
    emb_root: Path
    per_chain_n: Optional[int]
    seed: int
    train_ratio: float
    batch_size: int
    num_workers: int
    ast_vocab: Any
    cfg_vocab: Any


_CTX: Optional[FLContext] = None


def _resolve_chain_jsonl(data_root: Path, chain: str) -> Path:
    p1 = data_root / f"{chain}.jsonl"
    if p1.exists():
        return p1
    p2 = data_root / f"{chain}_500.jsonl"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find jsonl for chain={chain} under {data_root} (tried {p1.name}, {p2.name})")


def _resolve_emb_dir(emb_root: Path, chain: str) -> Path:
    p1 = emb_root / chain
    if p1.exists():
        return p1
    p2 = emb_root / f"{chain}_500"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find emb_dir for chain={chain} under {emb_root} (tried {p1.name}, {p2.name})")


def prepare_federated_context(
    chains: List[str],
    semantic_mode: str,
    data_root: str = "data/train/crossgraphnet_lite_labeled",
    emb_root: str = "data/embeddings",
    per_chain_n: Optional[int] = 500,
    seed: int = 42,
    train_ratio: float = 0.8,
    batch_size: int = 8,
    num_workers: int = 0,
) -> None:
    global _CTX

    if semantic_mode not in ("stats", "codebert_frozen"):
        raise ValueError(f"semantic_mode must be stats|codebert_frozen, got {semantic_mode}")

    data_root_p = Path(data_root)
    emb_root_p = Path(emb_root)

    merged_items = []
    for ch in chains:
        path = _resolve_chain_jsonl(data_root_p, ch)
        ds = MultiGraphJsonlDataset(path, limit=per_chain_n)
        merged_items.extend(ds.items)

    class _Tmp:
        def __init__(self, items):
            self.items = items

    ast_vocab, cfg_vocab = build_vocabs(_Tmp(merged_items))

    _CTX = FLContext(
        chains=list(chains),
        data_root=data_root_p,
        emb_root=emb_root_p,
        per_chain_n=per_chain_n,
        seed=seed,
        train_ratio=train_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        ast_vocab=ast_vocab,
        cfg_vocab=cfg_vocab,
    )


def build_model(semantic_mode: str, **kwargs) -> Any:
    if _CTX is None:
        raise RuntimeError("Call prepare_federated_context(...) before build_model().")

    if semantic_mode == "stats":
        sem_dim = 8
    elif semantic_mode == "codebert_frozen":
        sem_dim = 768
    else:
        raise ValueError(f"Unknown semantic_mode={semantic_mode}")

    cfg = CrossGraphNetLiteConfig(
        num_ast_types=len(_CTX.ast_vocab),
        num_cfg_types=len(_CTX.cfg_vocab),
        sem_dim=sem_dim,
        emb_dim=64,
        hidden_dim=64,
        num_classes=2,
        dropout=0.1,
    )
    return CrossGraphNetLite(cfg)


def build_loaders(chain: str, semantic_mode: str, **kwargs) -> Tuple[Any, Any, int]:
    if _CTX is None:
        raise RuntimeError("Call prepare_federated_context(...) before build_loaders().")

    per_chain_n = kwargs.get("per_chain_n", _CTX.per_chain_n)
    batch_size = kwargs.get("batch_size", _CTX.batch_size)
    num_workers = kwargs.get("num_workers", _CTX.num_workers)

    path = _resolve_chain_jsonl(_CTX.data_root, chain)
    full_ds = MultiGraphJsonlDataset(path, limit=per_chain_n)

    n_total = len(full_ds)
    n_train = int(n_total * _CTX.train_ratio)
    n_test = max(n_total - n_train, 1)

    g = torch.Generator().manual_seed(_CTX.seed)
    train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=g)

    if semantic_mode == "stats":
        sem_mode = "stats"
        emb_dir = None
    elif semantic_mode == "codebert_frozen":
        sem_mode = "llm"
        emb_dir = str(_resolve_emb_dir(_CTX.emb_root, chain))
    else:
        raise ValueError(f"Unknown semantic_mode={semantic_mode}")

    def _cf(b, sm=sem_mode, ed=emb_dir):
        return collate_fn(b, _CTX.ast_vocab, _CTX.cfg_vocab, sem_mode=sm, emb_dir=ed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_cf,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_cf,
    )
    return train_loader, test_loader, n_train


def train_one_epoch(
    model: Any,
    train_loader: Any,
    optimizer: Any,
    device: str = "cuda",
    algo: str = "fedavg",
    mu: float = 0.0,
    global_params=None,
) -> float:
    """
    FedAvg : loss = CE
    FedProx: loss = CE + (mu/2)*||theta - theta_global||^2
    """
    model.train()
    crit = nn.CrossEntropyLoss()
    losses = []

    use_prox = (algo.lower() == "fedprox") and (global_params is not None) and (mu is not None) and (mu > 0)

    for ast_b, cfg_b, sem, y in train_loader:
        ast_b = ast_b.to(device)
        cfg_b = cfg_b.to(device)
        sem = sem.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(
            ast_b.node_type, ast_b.edge_index, ast_b.batch,
            cfg_b.node_type, cfg_b.edge_index, cfg_b.batch,
            sem,
        )

        loss = crit(logits, y)

        if use_prox:
            prox = 0.0
            for p, p0 in zip(model.parameters(), global_params):
                prox = prox + torch.sum((p - p0) ** 2)
            loss = loss + 0.5 * float(mu) * prox

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())

    return float(np.mean(losses)) if losses else float("nan")


def evaluate(model: Any, test_loader: Any, device: str = "cuda") -> Dict[str, float]:
    return eval_crosschain(model, test_loader, device)
