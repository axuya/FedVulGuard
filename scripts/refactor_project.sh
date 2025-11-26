#!/usr/bin/env bash
set -e

echo "==== 开始一键重构 FedVulGuard 工程结构 ===="

# 创建新目录
mkdir -p src/collectors
mkdir -p src/preprocessors
mkdir -p src/dataset
mkdir -p src/eval
mkdir -p experiments
mkdir -p docs
mkdir -p data/graphs_raw
mkdir -p data/graphs_parquet
mkdir -p data/splits
mkdir -p data/spc

echo ">> 目录创建完成"

# 迁移 collectors
git mv src/data_collection/etherscan_crawler.py src/collectors/ 2>/dev/null || true
git mv src/data_collection/etherscan_crawler_5.py src/collectors/ 2>/dev/null || true
git mv src/data_collection/github_spc_crawler.py src/collectors/ 2>/dev/null || true
git mv src/data_collection/build_spc_from_datasets.py src/collectors/ 2>/dev/null || true
git mv src/data_collection/enhanced_spc_builder.py src/collectors/ 2>/dev/null || true

# 迁移 preprocessors
git mv src/preprocessing/build_graphs.py src/preprocessors/ 2>/dev/null || true
git mv src/preprocessing/simple_graph_builder.py src/preprocessors/ 2>/dev/null || true
git mv src/preprocessing/split_dataset.py src/preprocessors/ 2>/dev/null || true
git mv src/preprocessing/split_dataset_fixed.py src/preprocessors/ 2>/dev/null || true
git mv scripts/label_and_build_graphs.py src/preprocessors/ 2>/dev/null || true
git mv scripts/build_main_dataset_graphs.py src/preprocessors/ 2>/dev/null || true

# 迁移 dataset loader
git mv scripts/process_large.py src/dataset/ 2>/dev/null || true
git mv scripts/classify_unknown_types.py src/dataset/ 2>/dev/null || true

# 迁移 models
git mv src/models/train_mgvd.py src/models/ 2>/dev/null || true
git mv scripts/build_smartbugs_quick.py src/models/ 2>/dev/null || true
git mv scripts/label_slither_bulk.py src/models/ 2>/dev/null || true

# federated
git mv src/federated/train_federated.py src/federated/ 2>/dev/null || true

# eval
git mv scripts/verify_graph_data.py src/eval/ 2>/dev/null || true
git mv scripts/analyze_bootstrap_data.py src/eval/ 2>/dev/null || true
git mv scripts/filter_best_spc_pairs.py src/eval/ 2>/dev/null || true
git mv scripts/merge_all_spc.py src/eval/ 2>/dev/null || true

# 保留 scripts 里真正的脚本
mkdir -p scripts/backup
git mv scripts/run_all_rq_experiments.py scripts/ 2>/dev/null || true
git mv scripts/run_data_collection.py scripts/ 2>/dev/null || true
git mv scripts/setup_phase2_env.sh scripts/ 2>/dev/null || true

echo "==== 全部迁移动作完成 ===="
echo "接下来请执行: git add -A && git commit -m 'Refactor project structure'"

