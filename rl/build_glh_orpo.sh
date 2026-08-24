#!/usr/bin/env bash
# ORPO jsonl from matmul-only harvest pairs. Do not include conv.
# Pair list is a category filter of runs/glg_speed_pairs.jsonl.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIRS="${1:-$WS/runs/glh_speed_pairs.jsonl}"
OUT="${2:-$WS/runs/orpo_glh.jsonl}"

if [[ ! -s "$PAIRS" ]]; then
    echo "missing $PAIRS; filter glg_speed_pairs.jsonl to category=matmul" >&2
    exit 1
fi

cd "$FORGE"
export PYTHONPATH="$FORGE/kernelbench/src${PYTHONPATH:+:$PYTHONPATH}"
python3 train/build_orpo_dataset.py --pairs "$PAIRS" --out "$OUT"
