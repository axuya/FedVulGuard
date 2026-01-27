#!/usr/bin/env bash
set -euo pipefail

# Project root: scripts/ is under FedVulGuard/
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

LIST_DIR="$ROOT/results/sota_tools/lists"
OUT_DIR="$ROOT/results/sota_tools"

SMARTCHECK_IMAGE="${SMARTCHECK_IMAGE:-smartbugs/smartcheck:extended}"
SMARTCHECK_TIMEOUT="${SMARTCHECK_TIMEOUT:-600}"

SPLITS=("Ethereum_500" "BSC_500" "Fantom_500" "Polygon_500")

command -v docker >/dev/null 2>&1 || { echo "[ERR] docker not found."; exit 1; }
command -v timeout >/dev/null 2>&1 || { echo "[ERR] timeout not found. sudo apt-get install -y coreutils"; exit 1; }

mkdir -p "$OUT_DIR/smartcheck"

echo "========== [SmartCheck aligned] =========="
echo "[INFO] ROOT=$ROOT"
echo "[INFO] LIST_DIR=$LIST_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] SMARTCHECK_IMAGE=$SMARTCHECK_IMAGE"
echo "[INFO] SMARTCHECK_TIMEOUT=${SMARTCHECK_TIMEOUT}s"
echo "[INFO] SPLITS=${SPLITS[*]}"

run_smartcheck_full() {
  local tsv="$1"
  local tag="$2"
  mkdir -p "$OUT_DIR/smartcheck/$tag"

  local total=0
  while IFS=$'\t' read -r sid f; do
    [[ -z "${sid:-}" || -z "${f:-}" ]] && continue
    total=$((total+1))

    local out_txt="$OUT_DIR/smartcheck/$tag/${sid}.out.txt"
    local err_txt="$OUT_DIR/smartcheck/$tag/${sid}.err.txt"

    # RESUME
    if [[ -s "$out_txt" ]]; then
      continue
    fi

    if [[ ! -f "$f" ]]; then
      echo "[WARN][smartcheck][$tag] missing file: $f"
      continue
    fi

    # Convert absolute path under repo to container path /work/...
    local rel="${f#$ROOT/}"
    if [[ "$rel" == "$f" ]]; then
      echo "[WARN][smartcheck][$tag] file not under repo root, skip: $f"
      continue
    fi

    local tmp_out="${out_txt}.tmp"
    local tmp_err="${err_txt}.tmp"
    rm -f "$tmp_out" "$tmp_err" >/dev/null 2>&1 || true

    timeout "$SMARTCHECK_TIMEOUT" docker run --rm \
      -v "$ROOT:/work" \
      "$SMARTCHECK_IMAGE" \
      smartcheck -p "/work/${rel}" \
      1>"$tmp_out" 2>"$tmp_err" || true

    mv -f "$tmp_out" "$out_txt"

    # only keep stderr if non-empty
    if [[ -s "$tmp_err" ]]; then
      mv -f "$tmp_err" "$err_txt"
    else
      rm -f "$tmp_err" "$err_txt" >/dev/null 2>&1 || true
    fi

    if (( total % 25 == 0 )); then
      local done_cnt
      done_cnt="$(ls "$OUT_DIR/smartcheck/$tag" 2>/dev/null | wc -l || true)"
      echo "[INFO][smartcheck][$tag] processed=$total, outputs=$done_cnt"
    fi
  done < "$tsv"
}

for split in "${SPLITS[@]}"; do
  tsv="$LIST_DIR/${split}.tsv"
  [[ -f "$tsv" ]] || { echo "[ERR] missing tsv: $tsv (run run_sota_tools.sh first)"; exit 1; }

  echo "----- $split -----"
  echo "[STEP] smartcheck (FULL, aligned) -> $OUT_DIR/smartcheck/$split"
  run_smartcheck_full "$tsv" "$split"

  sc_cnt="$(ls "$OUT_DIR/smartcheck/$split" 2>/dev/null | wc -l || true)"
  echo "[DONE][$split] smartcheck_outputs=$sc_cnt"
done

echo "========== [DONE] =========="
echo "SmartCheck outputs: $OUT_DIR/smartcheck/<split>/<id>.out.txt"
echo "SmartCheck stderr:  only saved when non-empty (<id>.err.txt)"
