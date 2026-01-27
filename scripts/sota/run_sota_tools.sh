#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SOTA tools runner (Slither ONLY)
# - Exports id -> sol path TSV for each split (aligned 500)
# - Runs Slither FULL for each split
# - RESUME enabled
# - DOES NOT run Mythril (keeps existing mythril data untouched)
# ============================================================

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LIST_DIR="$ROOT/results/sota_tools/lists"
OUT_DIR="$ROOT/results/sota_tools"
JSONL_DIR="$ROOT/data/train/crossgraphnet_lite_labeled"
PY_EXPORT="$ROOT/scripts/sota/export_sol_by_id.py"

mkdir -p "$LIST_DIR" "$OUT_DIR"

SLITHER_IMAGE="smartbugs/slither:0.11.3"

# ✅ four aligned splits (500 each)
SPLITS=("Ethereum_500" "BSC_500" "Fantom_500" "Polygon_500")

# timeouts (seconds)
SLITHER_TIMEOUT="${SLITHER_TIMEOUT:-180}"

pick_shell() {
  local image="$1"
  if docker run --rm "$image" sh -lc 'echo ok' >/dev/null 2>&1; then
    echo "sh"
  elif docker run --rm "$image" bash -lc 'echo ok' >/dev/null 2>&1; then
    echo "bash"
  else
    echo ""
  fi
}

echo "========== [0] Preflight =========="
command -v docker >/dev/null 2>&1 || { echo "[ERR] docker not found."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERR] python3 not found."; exit 1; }
command -v timeout >/dev/null 2>&1 || { echo "[ERR] timeout not found. sudo apt-get install -y coreutils"; exit 1; }

echo "[INFO] ROOT=$ROOT"
echo "[INFO] LIST_DIR=$LIST_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] JSONL_DIR=$JSONL_DIR"
echo "[INFO] SLITHER_IMAGE=$SLITHER_IMAGE"
echo "[INFO] SLITHER_TIMEOUT=${SLITHER_TIMEOUT}s"
echo "[INFO] SPLITS=${SPLITS[*]}"
echo "[INFO] NOTE: Mythril is disabled in this script. Existing Mythril data under $OUT_DIR/mythril is kept as-is."

SLITHER_SHELL="$(pick_shell "$SLITHER_IMAGE")"
[[ -n "$SLITHER_SHELL" ]] || { echo "[ERR] no usable shell (sh/bash) in $SLITHER_IMAGE"; exit 1; }
echo "[INFO] SLITHER_SHELL=$SLITHER_SHELL"

echo "========== [1] Export id -> sol path =========="
for split in "${SPLITS[@]}"; do
  jsonl="$JSONL_DIR/${split}.jsonl"
  out_tsv="$LIST_DIR/${split}.tsv"
  [[ -f "$jsonl" ]] || { echo "[ERR] missing jsonl: $jsonl"; exit 1; }
  python3 "$PY_EXPORT" --test_jsonl "$jsonl" --out_tsv "$out_tsv" --root "$ROOT"
  echo "[OK] wrote list: $out_tsv ($(wc -l < "$out_tsv") rows)"
done

run_slither_full() {
  local tsv="$1"
  local tag="$2"
  mkdir -p "$OUT_DIR/slither/$tag"

  local total=0
  while IFS=$'\t' read -r sid f; do
    [[ -z "${sid:-}" || -z "${f:-}" ]] && continue
    total=$((total+1))

    local out_json="$OUT_DIR/slither/$tag/${sid}.json"
    if [[ -s "$out_json" ]]; then
      continue
    fi
    if [[ ! -f "$f" ]]; then
      echo "[WARN][slither][$tag] missing file: $f"
      continue
    fi

    local rel="${f#$ROOT/}"
    if [[ "$rel" == "$f" ]]; then
      echo "[WARN][slither][$tag] file not under repo root, skip: $f"
      continue
    fi

    # write to tmp then move (avoid empty/partial file on interruptions)
    local tmp_out="${out_json}.tmp"
    rm -f "$tmp_out" >/dev/null 2>&1 || true

    timeout "$SLITHER_TIMEOUT" docker run --rm \
      -v "$ROOT:/work" \
      "$SLITHER_IMAGE" \
      "$SLITHER_SHELL" -lc "slither /work/${rel} --json - --exclude-dependencies || true" \
      1>"$tmp_out" 2>/dev/null || true

    mv -f "$tmp_out" "$out_json"

    if (( total % 25 == 0 )); then
      local done_cnt
      done_cnt="$(ls "$OUT_DIR/slither/$tag" 2>/dev/null | wc -l || true)"
      echo "[INFO][slither][$tag] processed=$total, outputs=$done_cnt"
    fi
  done < "$tsv"
}

echo "========== [2] Run Slither (aligned, FULL) =========="
for split in "${SPLITS[@]}"; do
  tsv="$LIST_DIR/${split}.tsv"
  echo "----- $split -----"
  echo "[STEP] slither (FULL) -> $OUT_DIR/slither/$split"
  run_slither_full "$tsv" "$split"
  sl_cnt="$(ls "$OUT_DIR/slither/$split" 2>/dev/null | wc -l || true)"
  echo "[DONE][$split] slither_json=$sl_cnt"
done

echo "========== [DONE] =========="
echo "Slither outputs: $OUT_DIR/slither/<split>/<id>.json"
echo "Lists:           $LIST_DIR/<split>.tsv"
echo "NOTE: Mythril kept untouched at $OUT_DIR/mythril (script does not run it)."
