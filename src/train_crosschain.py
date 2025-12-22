from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from data_lite import MultiGraphJsonlDataset, build_vocabs, collate_fn
from model import CrossGraphNetLite, CrossGraphNetLiteConfig


def evaluate(model, loader, device):
    model.eval()
    ys, probs = [], []

    with torch.no_grad():
        for ast_b, cfg_b, sem, y in loader:
            ast_b = ast_b.to(device)
            cfg_b = cfg_b.to(device)
            sem = sem.to(device)
            y = y.to(device)

            logits = model(
                ast_b.node_type, ast_b.edge_index, ast_b.batch,
                cfg_b.node_type, cfg_b.edge_index, cfg_b.batch,
                sem,
            )
            p = torch.softmax(logits, dim=-1)[:, 1]

            ys.extend(y.detach().cpu().numpy().tolist())
            probs.extend(p.detach().cpu().numpy().tolist())

    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(int)

    return {
        "f1": float(f1_score(ys, preds)),
        "auc": float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else float("nan"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # ====== paths ======
    train_path = Path("data/train/crossgraphnet_lite_labeled/Ethereum.jsonl")
    test_path  = Path("data/train/crossgraphnet_lite_labeled/BSC_500.jsonl")

    # ====== semantic mode switch ======
    # choose: "stats" or "llm" or "none"
    SEM_MODE = "stats"
    EMB_DIR_TEST = "data/embeddings/BSC_500"

    # For stats mode, sem_dim must be 8; for llm mode, sem_dim must be 768; for none, sem_dim=0
    if SEM_MODE == "stats":
        sem_dim = 8
        emb_dir = None
    elif SEM_MODE == "llm":
        sem_dim = 768
        emb_dir = EMB_DIR_TEST
    else:
        sem_dim = 0
        emb_dir = None

    # ====== load datasets ======
    train_ds = MultiGraphJsonlDataset(train_path, limit=500)
    test_ds  = MultiGraphJsonlDataset(test_path,  limit=500)

    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")
    ast_vocab, cfg_vocab = build_vocabs(train_ds)
    print(f"AST vocab size: {len(ast_vocab)}")
    print(f"CFG vocab size: {len(cfg_vocab)}")

    # ====== loaders ======
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, ast_vocab, cfg_vocab, sem_mode=SEM_MODE, emb_dir=emb_dir),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, ast_vocab, cfg_vocab, sem_mode=SEM_MODE, emb_dir=emb_dir),
    )

    # ====== model ======
    cfg = CrossGraphNetLiteConfig(
        num_ast_types=len(ast_vocab),
        num_cfg_types=len(cfg_vocab),
        sem_dim=sem_dim,
        emb_dim=64,
        hidden_dim=64,
        num_classes=2,
        dropout=0.1,
    )
    model = CrossGraphNetLite(cfg).to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # ====== train ======
    epochs = 10
    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for ast_b, cfg_b, sem, y in train_loader:
            ast_b = ast_b.to(device)
            cfg_b = cfg_b.to(device)
            sem = sem.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(
                ast_b.node_type, ast_b.edge_index, ast_b.batch,
                cfg_b.node_type, cfg_b.edge_index, cfg_b.batch,
                sem,
            )
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()

        avg_loss = total / len(train_loader)
        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {ep} | loss={avg_loss:.4f} | [Test BSC] F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")
        model.train()

    # ====== gate inspect ======
    print("\n[Eval: inspecting gate behavior]")
    model.eval()
    with torch.no_grad():
        for ast_b, cfg_b, sem, y in test_loader:
            _ = model(
                ast_b.node_type.to(device), ast_b.edge_index.to(device), ast_b.batch.to(device),
                cfg_b.node_type.to(device), cfg_b.edge_index.to(device), cfg_b.batch.to(device),
                sem.to(device),
            )
            break

    # save minimal metrics snapshot (optional)
    out_dir = Path("results/experiments") / f"ETH_to_BSC_{SEM_MODE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.txt").open("w") as f:
        f.write(f"train={train_path}\n")
        f.write(f"test={test_path}\n")
        f.write(f"sem_mode={SEM_MODE}\n")
        f.write(f"sem_dim={sem_dim}\n")
        if emb_dir:
            f.write(f"emb_dir={emb_dir}\n")

    # write last metrics
    last = evaluate(model, test_loader, device)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(last, f, indent=2)
    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
