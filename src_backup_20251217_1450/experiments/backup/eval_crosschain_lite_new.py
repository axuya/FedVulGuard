import torch
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.experiments.train_crossgraphnet_lite import (
    CrossGraphNetLite,
    MultiGraphJsonlDataset,
    collate_fn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DEBUG 开关 ==========
DEBUG_BATCH = False   # 新对话第一步：设为 True 跑一次
# ==============================


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, preds = [], []

    for batch in loader:

        # -------- DEBUG：一次性打印 batch 真实结构 --------
        if DEBUG_BATCH:
            print("\n=== DEBUG BATCH STRUCTURE ===")
            print("Batch type:", type(batch))
            print("Batch len:", len(batch))
            for i, obj in enumerate(batch):
                print(f"\n[Item {i}] type={type(obj)}")
                if hasattr(obj, "edge_index"):
                    ei = obj.edge_index
                    print("  edge_index:", None if ei is None else tuple(ei.shape))
                if hasattr(obj, "x"):
                    x = obj.x
                    print("  x:", None if x is None else tuple(x.shape))
                if hasattr(obj, "batch"):
                    b = obj.batch
                    print("  batch:", None if b is None else tuple(b.shape))
                if isinstance(obj, torch.Tensor):
                    print("  tensor:", tuple(obj.shape), obj.dtype)
            raise SystemExit("DEBUG DONE")
        # ---------------------------------------------------

        # 约定：最后一个一定是 label
        *items, y = batch

        # label → tensor
        if isinstance(y[0], str):
            y = [int(v) if v.isdigit() else 1 for v in y]
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        # 收集图与 tensor
        graph_objs = []
        tensor_objs = []

        for obj in items:
            if hasattr(obj, "edge_index") and hasattr(obj, "batch"):
                graph_objs.append(obj.to(DEVICE))
            elif isinstance(obj, torch.Tensor):
                tensor_objs.append(obj.to(DEVICE))

        # ======== 核心假设（新对话将精确确认）========
        # graph_objs[0] → AST graph
        # graph_objs[1] → CFG graph
        ast_batch = graph_objs[0]
        cfg_batch = graph_objs[1]

        # AST type idx：来自 tensor_objs
        ast_type_idx = tensor_objs[0]

        # CFG type idx：优先用 cfg_batch.x
        cfg_type_idx = cfg_batch.x
        assert cfg_type_idx is not None, "CFG type idx missing"
        # ==============================================

        logits = model(
            ast_type_idx.long(),
            ast_batch.edge_index,
            ast_batch.batch,
            cfg_type_idx.long(),
            cfg_batch.edge_index,
            cfg_batch.batch,
        )

        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long()

        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        ys, preds, average="binary", zero_division=0
    )

    return {"acc": acc, "precision": p, "recall": r, "f1": f1}


def main():
    ckpt = torch.load("models/crossgraphnet_lite.pt", map_location=DEVICE)

    ast_vocab = ckpt["ast_vocab"]
    cfg_vocab = ckpt["cfg_vocab"]
    cfg = ckpt["config"]

    model = CrossGraphNetLite(
        num_ast_types=len(ast_vocab),
        num_cfg_types=len(cfg_vocab),
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("dropout", 0.1),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    collate = partial(
        collate_fn,
        ast_vocab=ast_vocab,
        cfg_vocab=cfg_vocab,
    )

    test_sets = {
        "Fantom": "data/train/crossgraphnet_lite_labeled/Fantom.jsonl",
        "Avalanche": "data/train/crossgraphnet_lite_labeled/Avalanche.jsonl",
    }

    for name, path in test_sets.items():
        dataset = MultiGraphJsonlDataset([Path(path)])
        loader = DataLoader(
            dataset,
            batch_size=cfg.get("batch_size", 32),
            shuffle=False,
            collate_fn=collate,
        )

        metrics = evaluate(model, loader)
        print(f"[{name}] {metrics}")


if __name__ == "__main__":
    main()
