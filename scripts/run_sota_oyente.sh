#!/usr/bin/env bash
set -euo pipefail

IN_ROOT=$(realpath ${IN_ROOT:-"data/raw"})
OUT_ROOT=${OUT_ROOT:-"results/sota_tools"}
CHAINS=${CHAINS:-"BSC Fantom Polygon"}
TIMEOUT_SEC=${TIMEOUT_SEC:-900}
N_LIMIT=${N_LIMIT:-0}
IMG=${IMG:-"smartbugs/oyente:480e725"}

run_one () {
  local chain="$1" base="$2"
  local in_dir="${IN_ROOT}/${chain}"
  local out_chain="${chain}_500"
  local out_dir="${OUT_ROOT}/oyente/${out_chain}"
  mkdir -p "${out_dir}"

  local out_file="${out_dir}/${base}.out.txt"
  local err_file="${out_dir}/${base}.err.txt"

  # checkpoint
  if [[ -s "${out_file}" || -s "${err_file}" ]]; then
    return 0
  fi

  timeout --kill-after=10 "${TIMEOUT_SEC}" \
    docker run --rm \
      -v "${in_dir}:/in:ro" \
      "${IMG}" \
      -s "/in/${base}" -ce \
    1>"${out_file}" 2>"${err_file}" || true
}

for chain in ${CHAINS}; do
  in_dir="${IN_ROOT}/${chain}"
  if [[ ! -d "${in_dir}" ]]; then
    echo "[WARN] missing input dir: ${in_dir}"
    continue
  fi

  echo "=============================="
  echo "[Oyente] ${chain} input=${in_dir}"
  echo "=============================="

  mapfile -t sols < <(find "${in_dir}" -maxdepth 1 -type f -name "*.sol" -printf "%f\n" | sort)
  if [[ "${N_LIMIT}" -gt 0 ]]; then
    sols=("${sols[@]:0:${N_LIMIT}}")
  fi

  for base in "${sols[@]}"; do
    run_one "${chain}" "${base}"
  done
done

echo "[DONE] Oyente finished"
