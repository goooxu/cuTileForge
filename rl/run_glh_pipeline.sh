#!/usr/bin/env bash
# Build matmul-only ORPO jsonl, train, merge GL-H.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED="${MERGED_OUT:-/raid/tmp/gemsg-cutile/model-GLH}"
mkdir -p "$WS/runs"
exec 8>"$WS/runs/.glh_pipeline.lock"
if ! flock -n 8; then
    echo "run_glh_pipeline already running"
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

echo "=== build ORPO jsonl ==="
if [[ ! -f "$WS/runs/orpo_glh.jsonl" ]]; then
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=none NAME=glh_mix \
        "$FORGE/kernelbench/scripts/in_container.sh" \
        "cd /ws/cuTileForge && CUTILE_WS=/ws bash rl/build_glh_orpo.sh \
            /ws/runs/glh_speed_pairs.jsonl /ws/runs/orpo_glh.jsonl"
fi
if [[ ! -s "$WS/runs/orpo_glh.jsonl" ]]; then
    echo "ORPO jsonl empty; not training"
    exit 1
fi

echo "=== ORPO + merge ==="
bash "$FORGE/rl/run_glh_orpo.sh"
echo "pipeline done: $MERGED"
