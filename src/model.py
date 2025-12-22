import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CrossGraphNetLiteConfig:
    def __init__(
        self,
        num_ast_types: int,
        num_cfg_types: int,
        sem_dim: int,              # 0/8/768
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        self.num_ast_types = num_ast_types
        self.num_cfg_types = num_cfg_types
        self.sem_dim = sem_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout


class GNNEncoderWithFallback(nn.Module):
    """
    If edge_index is empty -> bag-of-nodes fallback.
    """
    def __init__(self, num_types, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)
        self.bow_proj = nn.Linear(emb_dim, hidden_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_type, edge_index, batch):
        x = self.emb(x_type)

        if edge_index is None or edge_index.numel() == 0:
            out = torch.zeros(batch.max() + 1, x.size(-1), device=x.device)
            out.index_add_(0, batch, x)
            return self.dropout(self.bow_proj(out))

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        out = torch.zeros(batch.max() + 1, x.size(-1), device=x.device)
        out.index_add_(0, batch, x)
        return self.dropout(out)


class GatedFusion(nn.Module):
    """
    g = sigmoid(W[a;b])
    out = g*a + (1-g)*b
    """
    def __init__(self, hidden_dim: int, name: str):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, a, b):
        g = self.net(torch.cat([a, b], dim=-1))
        if not self.training:
            print(f"{self.name} gate mean:", g.mean().item())
        return g * a + (1.0 - g) * b


class CrossGraphNetLite(nn.Module):
    """
    AST-GNN + CFG-GNN + semantic vector -> two-level gated fusion
    """
    def __init__(self, cfg: CrossGraphNetLiteConfig):
        super().__init__()
        self.cfg = cfg

        self.ast_enc = GNNEncoderWithFallback(cfg.num_ast_types, cfg.emb_dim, cfg.hidden_dim, cfg.dropout)
        self.cfg_enc = GNNEncoderWithFallback(cfg.num_cfg_types, cfg.emb_dim, cfg.hidden_dim, cfg.dropout)

        # semantic projection (optional)
        if cfg.sem_dim and cfg.sem_dim > 0:
            self.sem_proj = nn.Sequential(
                nn.Linear(cfg.sem_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )
        else:
            self.sem_proj = None

        self.fuse_ast_cfg = GatedFusion(cfg.hidden_dim, "ast_cfg")
        self.fuse_struct = GatedFusion(cfg.hidden_dim, "struct_sem")

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(
        self,
        ast_type, ast_edge, ast_batch,
        cfg_type, cfg_edge, cfg_batch,
        struct_sem=None,  # [B, sem_dim] or None
    ):
        h_ast = self.ast_enc(ast_type, ast_edge, ast_batch)
        h_cfg = self.cfg_enc(cfg_type, cfg_edge, cfg_batch)
        h_struct = self.fuse_ast_cfg(h_ast, h_cfg)

        if self.sem_proj is None or struct_sem is None:
            # no semantic channel -> fuse with zeros
            h_sem = torch.zeros_like(h_struct)
        else:
            h_sem = self.sem_proj(struct_sem)

        h = self.fuse_struct(h_struct, h_sem)
        return self.classifier(h)
