#!/usr/bin/env bash
set -euo pipefail

# =========================
# Federated experiment runner
# =========================
# Assumes entrypoint: python -m src.federated.train_federated
# Logs: <LOGDIR>/<algo>_<semantic>_<tag>.jsonl
#
# Usage:
#   bash run_fl_exps.sh
#
# Optional environment overrides:
#   DEVICE=cuda LOGDIR=logs/fl DATA_ROOT=data/train/crossgraphnet_lite_labeled EMB_ROOT=data/embeddings \
#   CLIENTS="Ethereum BSC" bash run_fl_exps.sh
#
# Tip:
#   Start with RUN_MAIN=1 RUN_SENS_E=0 RUN_SENS_N=0 RUN_SENS_R=0 RUN_3CLIENT=0 for a quick pass.

DEVICE="${DEVICE:-cuda}"
LOGDIR="${LOGDIR:-logs/fl}"
DATA_ROOT="${DATA_ROOT:-data/train/crossgraphnet_lite_labeled}"
EMB_ROOT="${EMB_ROOT:-data/embeddings}"

# Default 2-client setup (cheapest, most stable)
CLIENTS_STR="${CLIENTS:-Ethereum BSC}"

TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# FedProx mu
MU="${MU:-0.001}"

# Base settings
BASE_N="${BASE_N:-500}"
BASE_E="${BASE_E:-1}"
BASE_R="${BASE_R:-10}"

# Seeds used in your paper-level reporting
SEEDS=(${SEEDS_OVERRIDE:-1 7 42})

# Semantics unified with your centralized sem_mode
SEMANTICS_MAIN=(none stats llm)

# Sensitivity sweeps (keep small; expand only if needed)
E_SWEEP=(1 2 5)
N_SWEEP=(500 1000)
R_SWEEP=(10 20 50)

# Switches (1=run, 0=skip)
RUN_MAIN="${RUN_MAIN:-1}"       # FedAvg/FedProx x semantics x seeds
RUN_SENS_E="${RUN_SENS_E:-1}"   # local_epochs sensitivity (seed42 by default)
RUN_SENS_N="${RUN_SENS_N:-1}"   # per_chain_n sensitivity (seed42 by default)
RUN_SENS_R="${RUN_SENS_R:-1}"   # rounds sensitivity (seed42 by default)
RUN_3CLIENT="${RUN_3CLIENT:-0}" # optional 3-client smoke + (optional) full seeds

# For sensitivity sweeps, default to seed42 unless overridden
SENS_SEEDS=(${SENS_SEEDS_OVERRIDE:-42})

# 3-client configs (edit as you like)
CLIENTS_3A=("Ethereum" "BSC" "Fantom")
CLIENTS_3B=("Ethereum" "BSC" "Polygon")

run_one () {
  local algo="$1"
  local semantic="$2"
  local tag="$3"
  local seed="$4"
  local per_chain_n="$5"
  local local_epochs="$6"
  local rounds="$7"
  local clients=("${@:8}")

  echo "============================================================"
  echo "RUN: algo=${algo} semantic=${semantic} seed=${seed} N=${per_chain_n} E=${local_epochs} R=${rounds}"
  echo "     clients=${clients[*]}"
  echo "     tag=${tag}"
  echo "============================================================"

  python -m src.federated.train_federated \
    --clients "${clients[@]}" \
    --semantic "${semantic}" \
    --algo "${algo}" \
    --mu "${MU}" \
    --rounds "${rounds}" \
    --local_epochs "${local_epochs}" \
    --device "${DEVICE}" \
    --per_chain_n "${per_chain_n}" \
    --train_ratio "${TRAIN_RATIO}" \
    --seed "${seed}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --data_root "${DATA_ROOT}" \
    --emb_root "${EMB_ROOT}" \
    --logdir "${LOGDIR}" \
    --tag "${tag}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}"
}

# Parse CLIENTS_STR into array
read -r -a CLIENTS_2 <<< "${CLIENTS_STR}"

# -------------------------
# 1) MAIN: FedAvg / FedProx x (none/stats/llm) x seeds
# -------------------------
if [[ "${RUN_MAIN}" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    for sem in "${SEMANTICS_MAIN[@]}"; do
      tag="C2_${CLIENTS_2[0]}${CLIENTS_2[1]}_${sem}_N${BASE_N}_E${BASE_E}_R${BASE_R}_seed${seed}"
      run_one "fedavg" "${sem}" "${tag}" "${seed}" "${BASE_N}" "${BASE_E}" "${BASE_R}" "${CLIENTS_2[@]}"
    done
  done

  # FedProx: usually most useful on stats/llm; keep none too if you want full table.
  for seed in "${SEEDS[@]}"; do
    for sem in "stats" "llm" "none"; do
      tag="C2_${CLIENTS_2[0]}${CLIENTS_2[1]}_FedProx_mu${MU}_${sem}_N${BASE_N}_E${BASE_E}_R${BASE_R}_seed${seed}"
      run_one "fedprox" "${sem}" "${tag}" "${seed}" "${BASE_N}" "${BASE_E}" "${BASE_R}" "${CLIENTS_2[@]}"
    done
  done
fi

# -------------------------
# 2) Sensitivity: local_epochs E (seed42 default)
# -------------------------
if [[ "${RUN_SENS_E}" == "1" ]]; then
  for seed in "${SENS_SEEDS[@]}"; do
    for sem in "llm"; do
      for e in "${E_SWEEP[@]}"; do
        tag="SENS_E_C2_${CLIENTS_2[0]}${CLIENTS_2[1]}_${sem}_N${BASE_N}_E${e}_R${BASE_R}_seed${seed}"
        run_one "fedavg" "${sem}" "${tag}" "${seed}" "${BASE_N}" "${e}" "${BASE_R}" "${CLIENTS_2[@]}"
      done
    done
  done
fi

# -------------------------
# 3) Sensitivity: sample size N (seed42 default)
# -------------------------
if [[ "${RUN_SENS_N}" == "1" ]]; then
  for seed in "${SENS_SEEDS[@]}"; do
    for sem in "llm"; do
      for n in "${N_SWEEP[@]}"; do
        tag="SENS_N_C2_${CLIENTS_2[0]}${CLIENTS_2[1]}_${sem}_N${n}_E${BASE_E}_R${BASE_R}_seed${seed}"
        run_one "fedavg" "${sem}" "${tag}" "${seed}" "${n}" "${BASE_E}" "${BASE_R}" "${CLIENTS_2[@]}"
      done
    done
  done
fi

# -------------------------
# 4) Sensitivity: rounds R (seed42 default)
# -------------------------
if [[ "${RUN_SENS_R}" == "1" ]]; then
  for seed in "${SENS_SEEDS[@]}"; do
    for sem in "stats"; do
      for r in "${R_SWEEP[@]}"; do
        tag="SENS_R_C2_${CLIENTS_2[0]}${CLIENTS_2[1]}_FedProx_mu${MU}_${sem}_N${BASE_N}_E${BASE_E}_R${r}_seed${seed}"
        run_one "fedprox" "${sem}" "${tag}" "${seed}" "${BASE_N}" "${BASE_E}" "${r}" "${CLIENTS_2[@]}"
      done
    done
  done
fi

# -------------------------
# 5) Optional: 3-client smoke tests (+ optional full seeds)
# -------------------------
if [[ "${RUN_3CLIENT}" == "1" ]]; then
  # Smoke test (seed42) for two 3-client settings
  for sem in "llm" "stats"; do
    tag="SMOKE_C3_EthBSCFantom_FedProx_mu${MU}_${sem}_N${BASE_N}_E${BASE_E}_R${BASE_R}_seed42"
    run_one "fedprox" "${sem}" "${tag}" "42" "${BASE_N}" "${BASE_E}" "${BASE_R}" "${CLIENTS_3A[@]}"

    tag="SMOKE_C3_EthBSCPolygon_FedProx_mu${MU}_${sem}_N${BASE_N}_E${BASE_E}_R${BASE_R}_seed42"
    run_one "fedprox" "${sem}" "${tag}" "42" "${BASE_N}" "${BASE_E}" "${BASE_R}" "${CLIENTS_3B[@]}"
  done

  # Uncomment if you want full 3-seed runs for 3-client configs
  # for seed in "${SEEDS[@]}"; do
  #   for sem in "llm" "stats"; do
  #     tag="C3_EthBSCFantom_FedProx_mu${MU}_${sem}_N${BASE_N}_E${BASE_E}_R${BASE_R}_seed${seed}"
  #     run_one "fedprox" "${sem}" "${tag}" "${seed}" "${BASE_N}" "${BASE_E}" "${BASE_R}" "${CLIENTS_3A[@]}"
  #
  #     tag="C3_EthBSCPolygon_FedProx_mu${MU}_${sem}_N${BASE_N}_E${BASE_E}_R${BASE_R}_seed${seed}"
  #     run_one "fedprox" "${sem}" "${tag}" "${seed}" "${BASE_N}" "${BASE_E}" "${BASE_R}" "${CLIENTS_3B[@]}"
  #   done
  # done
fi

echo "All requested runs finished. Logs are under: ${LOGDIR}"
