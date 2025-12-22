# CrossGraphNet-Lite

This repository contains a lightweight baseline for cross-chain smart contract vulnerability detection.

## Features
- AST-GNN + CFG-GNN dual structural encoders
- Gated Fusion for structure and semantics
- Graph-level statistics and frozen CodeBERT semantics
- Cross-chain generalization experiments

## Experiments
- Single-chain: Ethereum (500 samples)
- Cross-chain: Ethereum → BSC (500 / 500)

Key finding: frozen CodeBERT semantics significantly improves cross-chain generalization over graph-level statistics.

## How to Run
```bash
python src/train.py
python src/train_crosschain.py
