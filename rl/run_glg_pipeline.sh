#!/usr/bin/env bash
# Diagnose leftover is already written; build ORPO jsonl, train, merge GL-G.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED="${MERGED_OUT:-/raid/tmp/gemsg-cutile/model-GLG}"
mkdir -p "$WS/runs"
exec 8>"$WS/runs/.glg_pipeline.lock"
if ! flock -n 8; then
    echo "run_glg_pipeline already running"
    exit 0
fi

if [[ -f "$MERGED/processor_config.json" ]]; then
    echo "already merged $MERGED"
    exit 0
fi

if [[ ! -s "$WS/runs/glg_speed_pairs.jsonl" ]]; then
    echo "missing pair list; run rl/diagnose_speed_pairs.py" >&2
    exit 1
fi

echo "=== build ORPO jsonl ==="
if [[ ! -f "$WS/runs/orpo_glg.jsonl" ]]; then
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=none NAME=glg_mix \
        "$FORGE/kernelbench/scripts/in_container.sh" \
        "cd /ws/cuTileForge && CUTILE_WS=/ws bash rl/build_glg_orpo.sh \
            /ws/runs/glg_speed_pairs.jsonl /ws/runs/orpo_glg.jsonl"
fi
if [[ ! -s "$WS/runs/orpo_glg.jsonl" ]]; then
    echo "ORPO jsonl empty; not training"
    exit 1
fi

echo "=== ORPO + merge ==="
bash "$FORGE/rl/run_glg_orpo.sh"
echo "pipeline done: $MERGED"
