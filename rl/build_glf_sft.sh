#!/usr/bin/env bash
# Best-of-N speed SFT jsonl from the GL-E timed harvests.
# Keep only tasks that still lose to compile and already showed a wall-clock
# choice; one lowest-kernel_ms trace per problem. Do not mix distill.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$WS/runs/sft_glf.jsonl}"

cd "$FORGE"
export PYTHONPATH="$FORGE/kernelbench/src${PYTHONPATH:+:$PYTHONPATH}"
python3 train/build_sft_dataset.py \
    --run 86:$WS/runs/harvest_gle86:$WS/runs/harvest_gle86_verified.jsonl \
    --run 87:$WS/runs/harvest_gle87:$WS/runs/harvest_gle87_verified.jsonl \
    --run 92:$WS/runs/harvest_gle92:$WS/runs/harvest_gle92_verified.jsonl \
    --run 93:$WS/runs/harvest_gle93:$WS/runs/harvest_gle93_verified.jsonl \
    --completion-from response --prompt-tier cutile_concepts \
    --require-timing --max-per-problem 1 --rank-by kernel_ms \
    --min-timed-passes 2 --max-best-speedup 1.0 \
    --min-kernel-ms-spread 1.2 \
    --out "$OUT"
