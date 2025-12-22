from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class TypeVocab:
    """
    Stable node-type vocab.
    - Save/load via stoi dict to keep checkpoints portable.
    """
    def __init__(self):
        self.stoi: Dict[str, int] = {"<UNK>": 0}
        self.itos: List[str] = ["<UNK>"]

    def add(self, t: str) -> None:
        if t not in self.stoi:
            self.stoi[t] = len(self.itos)
            self.itos.append(t)

    def encode(self, t: str) -> int:
        return int(self.stoi.get(t, 0))

    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def from_stoi(stoi: Dict[str, int]) -> "TypeVocab":
        v = TypeVocab()
        v.stoi = {str(k): int(val) for k, val in stoi.items()}
        # rebuild itos (best-effort)
        max_id = max(v.stoi.values()) if v.stoi else 0
        itos = ["<UNK>"] * (max_id + 1)
        for tok, idx in v.stoi.items():
            if 0 <= idx < len(itos):
                itos[idx] = tok
        v.itos = itos
        if "<UNK>" not in v.stoi:
            v.stoi["<UNK>"] = 0
            if len(v.itos) == 0:
                v.itos = ["<UNK>"]
            v.itos[0] = "<UNK>"
        return v


class MultiGraphJsonlDataset(Dataset):
    """
    Expects JSONL items with:
      - label: int (0/1)
      - chain: str (optional)
      - graphs: {"ast": {...}, "cfg": {...}} (preferred)
        OR flat keys: ast_nodes/ast_edges/cfg_nodes/cfg_edges.
    """
    def __init__(self, paths: List[Path], chains_keep: Optional[set] = None):
        self.items: List[dict] = []
        for p in paths:
            with p.open("r", encoding="utf8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if "label" not in obj:
                        continue
                    chain = obj.get("chain", "Unknown")
                    if chains_keep is not None and chain not in chains_keep:
                        continue
                    self.items.append(obj)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _pick_ast_obj(obj: dict) -> dict:
    g = obj.get("graphs")
    if isinstance(g, dict):
        return g.get("ast") or {}
    return obj


def _pick_cfg_obj(obj: dict) -> dict:
    g = obj.get("graphs")
    if isinstance(g, dict):
        return g.get("cfg") or {}
    return obj


def build_vocabs(dataset: MultiGraphJsonlDataset, max_items: Optional[int] = None) -> Tuple[TypeVocab, TypeVocab]:
    ast_vocab, cfg_vocab = TypeVocab(), TypeVocab()
    n = len(dataset) if max_items is None else min(len(dataset), max_items)
    for i in range(n):
        obj = dataset[i]
        ast = _pick_ast_obj(obj)
        cfg = _pick_cfg_obj(obj)

        ast_nodes = ast.get("ast_nodes") or ast.get("nodes") or []
        for node in ast_nodes:
            t = node.get("type") if isinstance(node, dict) else None
            if t is not None:
                ast_vocab.add(str(t))

        cfg_nodes = cfg.get("cfg_nodes") or cfg.get("nodes") or []
        for node in cfg_nodes:
            t = node.get("type") if isinstance(node, dict) else None
            if t is not None:
                cfg_vocab.add(str(t))
    return ast_vocab, cfg_vocab


def _edges_to_edge_index(edges) -> torch.Tensor:
    """
    edges can be:
      - list of dicts: {"src": i, "dst": j} or {"u":..,"v":..}
      - list of pairs: [i, j]
    returns edge_index [2, E] long
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    src, dst = [], []
    for e in edges:
        if isinstance(e, dict):
            s = e.get("src", e.get("u", e.get("from")))
            d = e.get("dst", e.get("v", e.get("to")))
            if s is None or d is None:
                continue
            src.append(int(s)); dst.append(int(d))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            src.append(int(e[0])); dst.append(int(e[1]))
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def to_pyg_graph_ast(ast: dict, ast_vocab: TypeVocab) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes = ast.get("ast_nodes") or ast.get("nodes") or []
    x = torch.tensor([ast_vocab.encode(str(n.get("type", "<UNK>"))) for n in nodes if isinstance(n, dict)], dtype=torch.long)
    edges = ast.get("ast_edges") or ast.get("edges") or []
    edge_index = _edges_to_edge_index(edges)
    return x, edge_index


def to_pyg_graph_cfg(cfg: dict, cfg_vocab: TypeVocab) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes = cfg.get("cfg_nodes") or cfg.get("nodes") or []
    x = torch.tensor([cfg_vocab.encode(str(n.get("type", "<UNK>"))) for n in nodes if isinstance(n, dict)], dtype=torch.long)
    edges = cfg.get("cfg_edges") or cfg.get("edges") or []
    edge_index = _edges_to_edge_index(edges)
    return x, edge_index


def _ensure_nonempty_nodes(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((1,), dtype=torch.long)  # one UNK node
    return x


def _ensure_cfg_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # GraphSAGE can handle isolated nodes, but PyG ops sometimes expect some edges.
    # Provide at least one self-loop if empty.
    if edge_index.numel() == 0:
        if num_nodes <= 0:
            num_nodes = 1
        idx = torch.arange(num_nodes, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)
    return edge_index


def collate_fn(batch: List[dict], ast_vocab: TypeVocab, cfg_vocab: TypeVocab):
    ast_graphs, cfg_graphs, ys, chains = [], [], [], []
    for obj in batch:
        y = int(obj["label"])
        chain = obj.get("chain", "Unknown")

        ast = _pick_ast_obj(obj)
        cfg = _pick_cfg_obj(obj)

        ast_x, ast_edge = to_pyg_graph_ast(ast, ast_vocab)
        cfg_x, cfg_edge = to_pyg_graph_cfg(cfg, cfg_vocab)

        ast_x = _ensure_nonempty_nodes(ast_x)
        cfg_x = _ensure_nonempty_nodes(cfg_x)
        cfg_edge = _ensure_cfg_edges(cfg_edge, int(cfg_x.size(0)))

        ast_graphs.append(Data(node_type=ast_x, edge_index=ast_edge))
        cfg_graphs.append(Data(node_type=cfg_x, edge_index=cfg_edge))
        ys.append(y)
        chains.append(chain)

    ast_b = Batch.from_data_list(ast_graphs)
    cfg_b = Batch.from_data_list(cfg_graphs)
    y = torch.tensor(ys, dtype=torch.long)
    return ast_b, cfg_b, y, chains
