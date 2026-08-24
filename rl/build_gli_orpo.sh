#!/usr/bin/env bash
# Conv retain jsonl: harvest passes with no GEMM tile, one per problem.
# Pair list is the same 105 matmul rows as GL-H.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIRS="${1:-$WS/runs/glh_speed_pairs.jsonl}"
ORPO_OUT="${2:-$WS/runs/orpo_gli.jsonl}"
RETAIN_OUT="${3:-$WS/runs/sft_gli_retain.jsonl}"

if [[ ! -s "$PAIRS" ]]; then
    echo "missing $PAIRS" >&2
    exit 1
fi

cd "$FORGE"
export PYTHONPATH="$FORGE/kernelbench/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$ORPO_OUT" ]]; then
    python3 train/build_orpo_dataset.py --pairs "$PAIRS" --out "$ORPO_OUT"
fi

if [[ ! -f "$RETAIN_OUT" ]]; then
    python3 train/build_sft_dataset.py \
        --run "86:$WS/runs/harvest_gle86:$WS/runs/harvest_gle86_verified.jsonl" \
        --run "87:$WS/runs/harvest_gle87:$WS/runs/harvest_gle87_verified.jsonl" \
        --run "92:$WS/runs/harvest_gle92:$WS/runs/harvest_gle92_verified.jsonl" \
        --run "93:$WS/runs/harvest_gle93:$WS/runs/harvest_gle93_verified.jsonl" \
        --out "$RETAIN_OUT" \
        --prompt-tier cutile_concepts \
        --completion-from response \
        --max-per-problem 1 \
        --rank-by kernel_ms \
        --categories conv \
        --no-mma-tile
fi
