#!/usr/bin/env bash
# Build matmul ORPO + conv retain jsonl, train, merge GL-I from GL-E.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED="${MERGED_OUT:-/raid/tmp/gemsg-cutile/model-GLI}"
mkdir -p "$WS/runs"
exec 8>"$WS/runs/.gli_pipeline.lock"
if ! flock -n 8; then
    echo "run_gli_pipeline already running"
    exit 0
fi

if [[ -f "$MERGED/processor_config.json" ]]; then
    echo "already merged $MERGED"
    exit 0
fi

if [[ ! -s "$WS/runs/glh_speed_pairs.jsonl" ]]; then
    echo "missing $WS/runs/glh_speed_pairs.jsonl" >&2
    exit 1
fi

echo "=== build ORPO + retain jsonl ==="
if [[ -s "$WS/runs/orpo_gli.jsonl" && -s "$WS/runs/sft_gli_retain.jsonl" ]]; then
    echo "already have orpo_gli.jsonl and sft_gli_retain.jsonl"
else
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=none NAME=gli_mix \
        "$FORGE/kernelbench/scripts/in_container.sh" \
        "cd /ws/cuTileForge && CUTILE_WS=/ws bash rl/build_gli_orpo.sh \
            /ws/runs/glh_speed_pairs.jsonl /ws/runs/orpo_gli.jsonl \
            /ws/runs/sft_gli_retain.jsonl"
fi
if [[ ! -s "$WS/runs/orpo_gli.jsonl" || ! -s "$WS/runs/sft_gli_retain.jsonl" ]]; then
    echo "GL-I jsonl empty; not training"
    exit 1
fi

echo "=== ORPO + retain + merge ==="
bash "$FORGE/rl/run_gli_orpo.sh"
echo "pipeline done: $MERGED"
