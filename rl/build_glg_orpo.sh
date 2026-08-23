#!/usr/bin/env bash
# ORPO jsonl from the diagnosed GL-E harvest kernel_ms pairs.
# Diagnose first (rl/diagnose_speed_pairs.py); do not mix distill.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIRS="${1:-$WS/runs/glg_speed_pairs.jsonl}"
OUT="${2:-$WS/runs/orpo_glg.jsonl}"

if [[ ! -s "$PAIRS" ]]; then
    echo "missing $PAIRS; run rl/diagnose_speed_pairs.py first" >&2
    exit 1
fi

cd "$FORGE"
export PYTHONPATH="$FORGE/kernelbench/src${PYTHONPATH:+:$PYTHONPATH}"
python3 train/build_orpo_dataset.py --pairs "$PAIRS" --out "$OUT"
