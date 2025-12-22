import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
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


class TokenVocab:
    def __init__(self):
        self.stoi = {"<PAD>": 0, "<UNK>": 1}

    def add(self, t: str):
        if t not in self.stoi:
            self.stoi[t] = len(self.stoi)

    def encode(self, t: str) -> int:
        return self.stoi.get(t, 1)

    def __len__(self):
        return len(self.stoi)


class EthereumJsonlDataset(Dataset):
    def __init__(self, path: Path, limit: Optional[int] = None):
        self.items = []
        with path.open("r", encoding="utf8") as f:
            for line in f:
                if limit is not None and len(self.items) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "label" not in obj:
                    continue
                self.items.append(obj)

        if len(self.items) == 0:
            raise RuntimeError("No valid samples loaded")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_vocab(ds: EthereumJsonlDataset) -> Tuple[TypeVocab, TypeVocab, TokenVocab]:
    ast_vocab, cfg_vocab = TypeVocab(), TypeVocab()
    tok_vocab = TokenVocab()

    token_re = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|0x[0-9a-fA-F]+|\d+|==|!=|<=|>=|&&|\|\||[{}()\[\];,\.=\+\-\*/<>]")

    for obj in ds.items:
        ast = obj.get("graphs", {}).get("ast", obj)
        cfg = obj.get("graphs", {}).get("cfg", obj)

        for n in ast.get("nodes", []):
            ast_vocab.add(str(n.get("type", "<UNK>")))
        for n in cfg.get("nodes", []):
            cfg_vocab.add(str(n.get("type", "<UNK>")))

        # LLM-proxy tokens from source code
        code = obj.get("src_code", "") or obj.get("code", "") or ""
        if code:
            toks = token_re.findall(code)
            for t in toks[:512]:
                tok_vocab.add(t)

    return ast_vocab, cfg_vocab, tok_vocab


def build_edge_index(edges, num_nodes: int):
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    srcs, dsts = [], []
    for e in edges:
        if isinstance(e, dict):
            src, dst = e.get("src"), e.get("dst")
        else:
            src, dst = e[0], e[1]
        if src is None or dst is None:
            continue
        if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
            continue
        srcs.append(src)
        dsts.append(dst)

    if len(srcs) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([srcs, dsts], dtype=torch.long)


def tokenize_code(code: str, max_tokens: int = 128):
    token_re = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|0x[0-9a-fA-F]+|\d+|==|!=|<=|>=|&&|\|\||[{}()\[\];,\.=\+\-\*/<>]")
    toks = token_re.findall(code or "")
    return toks[:max_tokens]


def collate_fn(batch: List[dict], ast_vocab: TypeVocab, cfg_vocab: TypeVocab, tok_vocab: TokenVocab):
    ast_graphs, cfg_graphs, ys = [], [], []

    llm_token_ids = []
    llm_batch = []

    for i, obj in enumerate(batch):
        y = int(obj["label"])
        ast = obj.get("graphs", {}).get("ast", obj)
        cfg = obj.get("graphs", {}).get("cfg", obj)

        ast_nodes = ast.get("nodes", [])
        cfg_nodes = cfg.get("nodes", [])

        ast_x = torch.tensor(
            [ast_vocab.encode(str(n.get("type", "<UNK>"))) for n in ast_nodes],
            dtype=torch.long,
        )
        cfg_x = torch.tensor(
            [cfg_vocab.encode(str(n.get("type", "<UNK>"))) for n in cfg_nodes],
            dtype=torch.long,
        )

        if ast_x.numel() == 0:
            ast_x = torch.zeros((1,), dtype=torch.long)
        if cfg_x.numel() == 0:
            cfg_x = torch.zeros((1,), dtype=torch.long)

        ast_edge_index = build_edge_index(ast.get("edges", []), ast_x.size(0))

        ast_data = Data(node_type=ast_x, edge_index=ast_edge_index)
        cfg_data = Data(node_type=cfg_x, edge_index=torch.empty((2, 0), dtype=torch.long))

        ast_graphs.append(ast_data)
        cfg_graphs.append(cfg_data)
        ys.append(y)

        # LLM-proxy tokens
        code = obj.get("src_code", "") or obj.get("code", "") or ""
        toks = tokenize_code(code, max_tokens=128)
        if len(toks) == 0:
            toks = ["<UNK>"]
        ids = [tok_vocab.encode(t) for t in toks]
        llm_token_ids.extend(ids)
        llm_batch.extend([i] * len(ids))

    return (
        Batch.from_data_list(ast_graphs),
        Batch.from_data_list(cfg_graphs),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(llm_token_ids, dtype=torch.long),
        torch.tensor(llm_batch, dtype=torch.long),
    )
