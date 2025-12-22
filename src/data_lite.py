import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class TypeVocab:
    def __init__(self):
        self.stoi = {"<UNK>": 0}

    def add(self, t: str):
        if t not in self.stoi:
            self.stoi[t] = len(self.stoi)

    def encode(self, t: str) -> int:
        return self.stoi.get(t, 0)

    def __len__(self):
        return len(self.stoi)


class MultiGraphJsonlDataset(Dataset):
    def __init__(self, path: Path, limit: Optional[int] = None):
        self.items = []
        with path.open("r", encoding="utf8") as f:
            for line in f:
                if limit is not None and len(self.items) >= limit:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "label" not in obj:
                    continue
                self.items.append(obj)
        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples loaded from {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_vocabs(ds: MultiGraphJsonlDataset) -> Tuple[TypeVocab, TypeVocab]:
    ast_vocab = TypeVocab()
    cfg_vocab = TypeVocab()

    for obj in ds.items:
        graphs = obj.get("graphs", {})
        ast = graphs.get("ast", {}) or {}
        cfg = graphs.get("cfg", {}) or {}

        for n in ast.get("ast_nodes", []) or []:
            ast_vocab.add(str(n.get("type", "<UNK>")))

        for n in cfg.get("cfg_nodes", []) or []:
            cfg_vocab.add(str(n.get("type", "<UNK>")))


    return ast_vocab, cfg_vocab


def build_edge_index(edges, num_nodes: int) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    srcs, dsts = [], []
    for e in edges:
        if isinstance(e, dict):
            s, d = e.get("src"), e.get("dst")
        else:
            s, d = e[0], e[1]
        if s is None or d is None:
            continue
        if not (0 <= s < num_nodes and 0 <= d < num_nodes):
            continue
        srcs.append(s)
        dsts.append(d)

    if len(srcs) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([srcs, dsts], dtype=torch.long)


class LLMEmbeddingStore:
    def __init__(self, emb_dir: Path):
        ids_path = emb_dir / "ids.txt"
        emb_path = emb_dir / "emb.npy"
        assert ids_path.exists(), f"Missing {ids_path}"
        assert emb_path.exists(), f"Missing {emb_path}"

        with ids_path.open("r") as f:
            ids = [line.strip() for line in f]

        embs = np.load(emb_path)
        assert len(ids) == embs.shape[0], "ID / embedding size mismatch"

        self.dim = embs.shape[1]
        self.id2idx = {cid: i for i, cid in enumerate(ids)}
        self.embs = torch.from_numpy(embs).float()

    def get(self, cid: str) -> torch.Tensor:
        idx = self.id2idx.get(cid, None)
        if idx is None:
            return torch.zeros(self.dim)
        return self.embs[idx]


def _stats_sem(ast_nodes, ast_edges, cfg_nodes, cfg_edges) -> torch.Tensor:
    # 8-d graph-level statistics
    na = len(ast_nodes) if ast_nodes else 0
    nc = len(cfg_nodes) if cfg_nodes else 0
    ea = len(ast_edges) if ast_edges else 0
    ec = len(cfg_edges) if cfg_edges else 0

    avg_deg_a = (ea / max(na, 1))
    avg_deg_c = (ec / max(nc, 1))
    node_ratio = (na / max(nc, 1))
    edge_ratio = (ea / max(ec, 1))
    has_cfg = 1.0 if ec > 0 else 0.0
    has_ast = 1.0 if ea > 0 else 0.0

    return torch.tensor(
        [na, nc, ea, ec, avg_deg_a, avg_deg_c, node_ratio, has_cfg],
        dtype=torch.float,
    )


def collate_fn(
    batch: List[dict],
    ast_vocab: TypeVocab,
    cfg_vocab: TypeVocab,
    sem_mode: str = "llm",              # "llm" | "stats" | "none"
    emb_dir: Optional[str] = None,      # required if sem_mode="llm"
):
    if sem_mode not in ("llm", "stats", "none"):
        raise ValueError(f"Unknown sem_mode={sem_mode}")

    # cache embedding store by emb_dir
    llm_store = None
    if sem_mode == "llm":
        if emb_dir is None:
            raise ValueError("emb_dir is required when sem_mode='llm'")
        key = f"_llm_store::{emb_dir}"
        if not hasattr(collate_fn, key):
            setattr(collate_fn, key, LLMEmbeddingStore(Path(emb_dir)))
        llm_store = getattr(collate_fn, key)

    ast_graphs, cfg_graphs, labels, sem_feats = [], [], [], []

    for obj in batch:
        labels.append(int(obj["label"]))
        cid = obj.get("id", "")

        graphs = obj.get("graphs", {})
        ast = graphs.get("ast", {}) or {}
        cfg = graphs.get("cfg", {}) or {}

        # === FIX: real schema ===
        ast_nodes = ast.get("ast_nodes", []) or []
        ast_edges = []  # Lite version: AST edges not provided here

        cfg_nodes = cfg.get("cfg_nodes", []) or []
        cfg_edges = cfg.get("cfg_edges", []) or []


        # AST graph
        ast_x = torch.tensor([ast_vocab.encode(str(n.get("type", "<UNK>"))) for n in ast_nodes], dtype=torch.long)
        if ast_x.numel() == 0:
            ast_x = torch.zeros((1,), dtype=torch.long)
        ast_edge_index = build_edge_index(ast_edges, ast_x.size(0))
        ast_graphs.append(Data(node_type=ast_x, edge_index=ast_edge_index))

        # CFG graph
        cfg_x = torch.tensor([cfg_vocab.encode(str(n.get("type", "<UNK>"))) for n in cfg_nodes], dtype=torch.long)
        if cfg_x.numel() == 0:
            cfg_x = torch.zeros((1,), dtype=torch.long)
        cfg_edge_index = build_edge_index(cfg_edges, cfg_x.size(0))
        cfg_graphs.append(Data(node_type=cfg_x, edge_index=cfg_edge_index))

        # semantic feature
        if sem_mode == "none":
            sem_feats.append(torch.zeros(1))  # placeholder; model will ignore if sem_dim=0
        elif sem_mode == "stats":
            sem_feats.append(_stats_sem(ast_nodes, ast_edges, cfg_nodes, cfg_edges))
        else:
            sem_feats.append(llm_store.get(cid))

    ast_batch = Batch.from_data_list(ast_graphs)
    cfg_batch = Batch.from_data_list(cfg_graphs)

    return (
        ast_batch,
        cfg_batch,
        torch.stack(sem_feats, dim=0),
        torch.tensor(labels, dtype=torch.long),
    )
