from pathlib import Path
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

#from data_lite import MultiGraphJsonlDataset, build_vocabs, collate_fn
#from model import CrossGraphNetLite, CrossGraphNetLiteConfig

from src.data_lite import MultiGraphJsonlDataset,build_vocabs,collate_fn
from src.model import CrossGraphNetLite,CrossGraphNetLiteConfig##加入联邦学习后修改的


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
    parser = argparse.ArgumentParser(description="CrossGraphNet-Lite cross-chain training (runs stats + llm in one go).")
    parser.add_argument("--train_path", type=str, default="data/train/crossgraphnet_lite_labeled/Ethereum.jsonl")
    parser.add_argument("--test_path", type=str, default="data/train/crossgraphnet_lite_labeled/BSC_500.jsonl")
    parser.add_argument("--emb_dir_test", type=str, default="data/embeddings/BSC_500",
                        help="Embedding directory for the TEST chain (used when sem_mode=llm).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_root", type=str, default="results/experiments",
                        help="Root directory to store outputs.")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ====== paths ======
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)

    # ====== dataset (graph only; semantic loaded in collate) ======
    train_ds = MultiGraphJsonlDataset(train_path)
    test_ds = MultiGraphJsonlDataset(test_path)

    print(f"Train: {train_path}  (n={len(train_ds)})")
    print(f"Test : {test_path}   (n={len(test_ds)})")

    # Build vocabs ONCE from training set (shared across modes)
    ast_vocab, cfg_vocab = build_vocabs(train_ds)
    print(f"AST vocab size: {len(ast_vocab)}")
    print(f"CFG vocab size: {len(cfg_vocab)}")

    # ====== run matrix ======
    MODES = ["stats", "llm"]
    results_root = Path(args.out_root) / "crosschain_runs"
    results_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_path_csv = results_root / "summary.csv"
    summary_path_jsonl = results_root / "summary.jsonl"

    for sem_mode in MODES:
        if sem_mode == "stats":
            sem_dim = 8
            emb_dir = None
        elif sem_mode == "llm":
            sem_dim = 768
            emb_dir = args.emb_dir_test
        else:
            raise ValueError(f"Unknown sem_mode: {sem_mode}")

        run_name = f"{train_path.stem}_to_{test_path.stem}_{sem_mode}_seed{args.seed}"
        out_dir = results_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # loaders (mode-specific collate)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b, sm=sem_mode, ed=emb_dir: collate_fn(b, ast_vocab, cfg_vocab, sem_mode=sm, emb_dir=ed),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b, sm=sem_mode, ed=emb_dir: collate_fn(b, ast_vocab, cfg_vocab, sem_mode=sm, emb_dir=ed),
        )

        # model
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
        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()

        # log config
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                    "sem_mode": sem_mode,
                    "sem_dim": sem_dim,
                    "emb_dir_test": emb_dir,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "device": str(device),
                },
                f,
                indent=2,
            )

        print(f"\n===== RUN: {run_name} =====")
        best = {"epoch": -1, "f1": -1.0, "auc": float("nan"), "loss": float("inf")}

        # training
        for ep in range(1, args.epochs + 1):
            model.train()
            losses = []

            for ast_b, cfg_b, sem, y in train_loader:
                ast_b = ast_b.to(device)
                cfg_b = cfg_b.to(device)
                sem = sem.to(device)
                y = y.to(device)

                optim.zero_grad()
                logits = model(
                    ast_b.node_type, ast_b.edge_index, ast_b.batch,
                    cfg_b.node_type, cfg_b.edge_index, cfg_b.batch,
                    sem,
                )
                loss = crit(logits, y)
                loss.backward()
                optim.step()
                losses.append(loss.detach().cpu().item())

            avg_loss = float(np.mean(losses)) if losses else float("nan")
            test_metrics = evaluate(model, test_loader, device)

            row = {
                "run": run_name,
                "mode": sem_mode,
                "epoch": ep,
                "train_loss": avg_loss,
                "test_f1": test_metrics["f1"],
                "test_auc": test_metrics["auc"],
            }
            summary_rows.append(row)

            if test_metrics["f1"] > best["f1"]:
                best = {"epoch": ep, "f1": test_metrics["f1"], "auc": test_metrics["auc"], "loss": avg_loss}
                # Save best checkpoint
                torch.save(model.state_dict(), out_dir / "model_best.pt")

            print(
                f"Epoch {ep:02d} | loss={avg_loss:.4f} | "
                f"[Test {test_path.stem}] F1={test_metrics['f1']:.4f} AUC={test_metrics['auc']:.4f}"
            )

        # Save last checkpoint + best metrics
        torch.save(model.state_dict(), out_dir / "model_last.pt")
        with (out_dir / "metrics_best.json").open("w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        print(f"Saved run to: {out_dir}")

    # ====== write combined summary (single document) ======
    import csv
    with summary_path_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['run','mode','epoch','train_loss','test_f1','test_auc'])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    with summary_path_jsonl.open("w", encoding="utf-8") as f:
        for r in summary_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nAll done. Combined summary written to:\n- {summary_path_csv}\n- {summary_path_jsonl}\n")


if __name__ == "__main__":
    main()
