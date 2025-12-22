from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_lite import (
    MultiGraphJsonlDataset,
    build_vocabs,
    collate_fn,
)

from model import CrossGraphNetLite, CrossGraphNetLiteConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # -----------------------------
    # Dataset
    # -----------------------------
    data_path = Path("data/train/crossgraphnet_lite_labeled/Ethereum.jsonl")
    ds = MultiGraphJsonlDataset(data_path, limit=500)

    print(f"Loaded dataset size: {len(ds)}")

    ast_vocab, cfg_vocab = build_vocabs(ds)
    print(f"AST vocab size: {len(ast_vocab)}")
    print(f"CFG vocab size: {len(cfg_vocab)}")

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, ast_vocab, cfg_vocab),
    )

    # -----------------------------
    # Model (LLM semantic version)
    # -----------------------------
    cfg = CrossGraphNetLiteConfig(
        num_ast_types=len(ast_vocab),
        num_cfg_types=len(cfg_vocab),
        sem_dim=768,          # 👈 CodeBERT
        emb_dim=64,
        hidden_dim=64,
        num_classes=2,
        dropout=0.1,
    )

    model = CrossGraphNetLite(cfg).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # Training
    # -----------------------------
    model.train()
    epochs = 10

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for ast_b, cfg_b, struct_sem, y in loader:
            ast_b = ast_b.to(device)
            cfg_b = cfg_b.to(device)
            struct_sem = struct_sem.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(
                ast_b.node_type,
                ast_b.edge_index,
                ast_b.batch,
                cfg_b.node_type,
                cfg_b.batch,
                struct_sem,
            )

            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | loss={total_loss / len(loader):.4f}")

    # -----------------------------
    # Eval: inspect gates
    # -----------------------------
    print("\n[Eval: inspecting gate behavior]")
    model.eval()
    with torch.no_grad():
        for ast_b, cfg_b, struct_sem, y in loader:
            _ = model(
                ast_b.node_type.to(device),
                ast_b.edge_index.to(device),
                ast_b.batch.to(device),
                cfg_b.node_type.to(device),
                cfg_b.batch.to(device),
                struct_sem.to(device),
            )
            break


if __name__ == "__main__":
    main()
