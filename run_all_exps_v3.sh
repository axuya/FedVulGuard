#!/usr/bin/env bash
set -euo pipefail

# Always run from the directory where this script lives (project root).
cd "$(dirname "$0")"

# ============================================================
# All Experiments Runner (v3): Centralized + Federated
# ============================================================
# Centralized: python -m src.train_crosschain
# Federated  : python -m src.federated.train_federated
#
# Targets: Ethereum -> {BSC, Fantom, Polygon}
# Semantics: none|stats|llm (FL side; centralized depends on your train_crosschain implementation)
# Seeds: 1,7,42
# FedProx mu scan: {0, 1e-4, 1e-3, 1e-2} (mu=0 => FedAvg baseline)
# ============================================================

# -------------------------
# Environment / Paths
# -------------------------
DEVICE="${DEVICE:-cuda}"

# Data roots
DATA_DIR="${DATA_DIR:-data/train/crossgraphnet_lite_labeled}"
EMB_DIR="${EMB_DIR:-data/embeddings}"

# Output roots
CENTRAL_OUT_ROOT="${CENTRAL_OUT_ROOT:-results/experiments}"
FL_LOGDIR="${FL_LOGDIR:-logs/fl}"

# Central training config
CENTRAL_EPOCHS="${CENTRAL_EPOCHS:-10}"
CENTRAL_BS="${CENTRAL_BS:-8}"

# Federated training config
FL_TRAIN_RATIO="${FL_TRAIN_RATIO:-0.8}"
FL_BS="${FL_BS:-8}"
FL_NUM_WORKERS="${FL_NUM_WORKERS:-0}"
FL_LR="${FL_LR:-1e-3}"
FL_WD="${FL_WD:-0.0}"

# Base FL setup
FL_BASE_N="${FL_BASE_N:-500}"
FL_BASE_E="${FL_BASE_E:-1}"
FL_BASE_R="${FL_BASE_R:-10}"

# Default FedProx mu (used in FL main runs)
MU="${MU:-0.001}"

# Seeds (paper-grade default)
SEEDS=(${SEEDS_OVERRIDE:-1 7 42})

# Semantic modes (FL side)
SEMANTICS=(none stats llm)

# Mu scan grid (mu=0 => FedAvg)
MU_GRID=(${MU_GRID_OVERRIDE:-0 1e-4 1e-3 1e-2})

# Sensitivity toggles
RUN_CENTRAL="${RUN_CENTRAL:-1}"
RUN_FL_MAIN="${RUN_FL_MAIN:-1}"
RUN_FL_MU_SCAN="${RUN_FL_MU_SCAN:-1}"
RUN_FL_SENS_E="${RUN_FL_SENS_E:-0}"
RUN_FL_SENS_N="${RUN_FL_SENS_N:-0}"
RUN_FL_SENS_R="${RUN_FL_SENS_R:-0}"

# Sensitivity settings (trend-first default seed42)
SENS_SEEDS=(${SENS_SEEDS_OVERRIDE:-42})
E_SWEEP=(${E_SWEEP_OVERRIDE:-1 2 5})
N_SWEEP=(${N_SWEEP_OVERRIDE:-500 1000})
R_SWEEP=(${R_SWEEP_OVERRIDE:-10 20 50})
SENS_SEM="${SENS_SEM:-llm}"          # llm or stats
MU_SCAN_SEM="${MU_SCAN_SEM:-stats}"  # stats recommended

mkdir -p "${CENTRAL_OUT_ROOT}" "${FL_LOGDIR}"

# -------------------------
# Cross-chain targets
# -------------------------
# Assumes:
#   Train: Ethereum.jsonl
#   Test : <Chain>_500.jsonl
#   Emb  : <Chain>_500/
TARGETS=("BSC" "Fantom" "Polygon")

# -------------------------
# Helpers: dataset paths
# -------------------------
train_jsonl () {
  echo "${DATA_DIR}/Ethereum.jsonl"
}

test_jsonl () {
  local chain="$1"
  echo "${DATA_DIR}/${chain}_500.jsonl"
}

emb_dir_test () {
  local chain="$1"
  echo "${EMB_DIR}/${chain}_500"
}

# -------------------------
# Centralized runner
# -------------------------
run_central_one () {
  local target="$1"
  local seed="$2"

  local train_path
  train_path="$(train_jsonl)"
  local test_path
  test_path="$(test_jsonl "${target}")"
  local emb
  emb="$(emb_dir_test "${target}")"

  echo "============================================================"
  echo "[CENTRAL] Ethereum -> ${target} | seed=${seed}"
  echo "  train_path=${train_path}"
  echo "  test_path=${test_path}"
  echo "  emb_dir_test=${emb}"
  echo "  epochs=${CENTRAL_EPOCHS} batch_size=${CENTRAL_BS}"
  echo "  out_root=${CENTRAL_OUT_ROOT}"
  echo "============================================================"

  python -m src.train_crosschain \
    --train_path "${train_path}" \
    --test_path "${test_path}" \
    --emb_dir_test "${emb}" \
    --epochs "${CENTRAL_EPOCHS}" \
    --batch_size "${CENTRAL_BS}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --out_root "${CENTRAL_OUT_ROOT}"
}

# -------------------------
# Federated runner
# -------------------------
run_fl_one () {
  local client_a="$1"
  local client_b="$2"
  local algo="$3"
  local semantic="$4"
  local tag="$5"
  local seed="$6"
  local n="$7"
  local e="$8"
  local r="$9"
  local mu="${10}"

  echo "============================================================"
  echo "[FL] clients=${client_a}+${client_b} algo=${algo} sem=${semantic} seed=${seed} N=${n} E=${e} R=${r} mu=${mu}"
  echo "     tag=${tag}"
  echo "============================================================"

  python -m src.federated.train_federated \
    --clients "${client_a}" "${client_b}" \
    --semantic "${semantic}" \
    --algo "${algo}" \
    --mu "${mu}" \
    --rounds "${r}" \
    --local_epochs "${e}" \
    --device "${DEVICE}" \
    --per_chain_n "${n}" \
    --train_ratio "${FL_TRAIN_RATIO}" \
    --seed "${seed}" \
    --batch_size "${FL_BS}" \
    --num_workers "${FL_NUM_WORKERS}" \
    --data_root "${DATA_DIR}" \
    --emb_root "${EMB_DIR}" \
    --logdir "${FL_LOGDIR}" \
    --tag "${tag}" \
    --lr "${FL_LR}" \
    --weight_decay "${FL_WD}"
}

# ============================================================
# 1) CENTRALIZED: Eth -> {BSC,Fantom,Polygon} x seeds
# ============================================================
if [[ "${RUN_CENTRAL}" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    for t in "${TARGETS[@]}"; do
      run_central_one "${t}" "${seed}"
    done
  done
fi

# ============================================================
# 2) FEDERATED MAIN: (Eth+Target) x (FedAvg/FedProx) x semantics x seeds
# ============================================================
if [[ "${RUN_FL_MAIN}" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    for t in "${TARGETS[@]}"; do
      # FedAvg main
      for sem in "${SEMANTICS[@]}"; do
        tag="C2_Eth${t}_fedavg_${sem}_N${FL_BASE_N}_E${FL_BASE_E}_R${FL_BASE_R}_seed${seed}"
        run_fl_one "Ethereum" "${t}" "fedavg" "${sem}" "${tag}" "${seed}" "${FL_BASE_N}" "${FL_BASE_E}" "${FL_BASE_R}" "0.0"
      done

      # FedProx main (default MU)
      for sem in "${SEMANTICS[@]}"; do
        tag="C2_Eth${t}_fedprox_mu${MU}_${sem}_N${FL_BASE_N}_E${FL_BASE_E}_R${FL_BASE_R}_seed${seed}"
        run_fl_one "Ethereum" "${t}" "fedprox" "${sem}" "${tag}" "${seed}" "${FL_BASE_N}" "${FL_BASE_E}" "${FL_BASE_R}" "${MU}"
      done
    done
  done
fi

# ============================================================
# 3) FEDPROX MU SCAN: per target, per mu (seed42 trend by default)
# ============================================================
if [[ "${RUN_FL_MU_SCAN}" == "1" ]]; then
  R_MU_SCAN="${R_MU_SCAN_OVERRIDE:-20}"
  for seed in "${SENS_SEEDS[@]}"; do
    for t in "${TARGETS[@]}"; do
      for mu in "${MU_GRID[@]}"; do
        if [[ "${mu}" == "0" || "${mu}" == "0.0" ]]; then
          tag="MUscan_Eth${t}_${MU_SCAN_SEM}_mu0_N${FL_BASE_N}_E${FL_BASE_E}_R${R_MU_SCAN}_seed${seed}"
          run_fl_one "Ethereum" "${t}" "fedavg" "${MU_SCAN_SEM}" "${tag}" "${seed}" "${FL_BASE_N}" "${FL_BASE_E}" "${R_MU_SCAN}" "0.0"
        else
          tag="MUscan_Eth${t}_${MU_SCAN_SEM}_mu${mu}_N${FL_BASE_N}_E${FL_BASE_E}_R${R_MU_SCAN}_seed${seed}"
          run_fl_one "Ethereum" "${t}" "fedprox" "${MU_SCAN_SEM}" "${tag}" "${seed}" "${FL_BASE_N}" "${FL_BASE_E}" "${R_MU_SCAN}" "${mu}"
        fi
      done
    done
  done
fi

# ============================================================
# 4) Sensitivity (optional): E / N / R (trend-first default seed42)
# ============================================================
if [[ "${RUN_FL_SENS_E}" == "1" ]]; then
  for seed in "${SENS_SEEDS[@]}"; do
    for t in "${TARGETS[@]}"; do
      for e in "${E_SWEEP[@]}"; do
        tag="SENS_E_Eth${t}_${SENS_SEM}_N${FL_BASE_N}_E${e}_R${FL_BASE_R}_seed${seed}"
        run_fl_one "Ethereum" "${t}" "fedavg" "${SENS_SEM}" "${tag}" "${seed}" "${FL_BASE_N}" "${e}" "${FL_BASE_R}" "0.0"
      done
    done
  done
fi

if [[ "${RUN_FL_SENS_N}" == "1" ]]; then
  for seed in "${SENS_SEEDS[@]}"; do
    for t in "${TARGETS[@]}"; do
      for n in "${N_SWEEP[@]}"; do
        tag="SENS_N_Eth${t}_${SENS_SEM}_N${n}_E${FL_BASE_E}_R${FL_BASE_R}_seed${seed}"
        run_fl_one "Ethereum" "${t}" "fedavg" "${SENS_SEM}" "${tag}" "${seed}" "${n}" "${FL_BASE_E}" "${FL_BASE_R}" "0.0"
      done
    done
  done
fi

if [[ "${RUN_FL_SENS_R}" == "1" ]]; then
  for seed in "${SENS_SEEDS[@]}"; do
    for t in "${TARGETS[@]}"; do
      for r in "${R_SWEEP[@]}"; do
        tag="SENS_R_Eth${t}_fedprox_mu${MU}_${MU_SCAN_SEM}_N${FL_BASE_N}_E${FL_BASE_E}_R${r}_seed${seed}"
        run_fl_one "Ethereum" "${t}" "fedprox" "${MU_SCAN_SEM}" "${tag}" "${seed}" "${FL_BASE_N}" "${FL_BASE_E}" "${r}" "${MU}"
      done
    done
  done
fi

echo "============================================================"
echo "DONE."
echo "  [CENTRAL] results under: ${CENTRAL_OUT_ROOT} (see crosschain_runs/summary.csv,jsonl from src.train_crosschain)"
echo "  [FL]      logs under:    ${FL_LOGDIR}/*.jsonl"
echo "Next:"
echo "  - FL summary: bash summarize_fl.sh   (or LOGDIR='.' OUTDIR='./_summary' bash summarize_fl.sh inside ${FL_LOGDIR})"
echo "============================================================"
