#!/usr/bin/env bash
# After GL-E timed harvests finish: gate, best-of-N jsonl, SFT, merge.
# Exit 2 while harvests are still running. Exit 1 if the speed gate fails.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED="${MERGED_OUT:-/raid/tmp/gemsg-cutile/model-GLF}"
mkdir -p "$WS/runs"
exec 8>"$WS/runs/.glf_pipeline.lock"
if ! flock -n 8; then
    echo "run_glf_pipeline already running"
    exit 0
fi

need=(
    "$WS/runs/harvest_gle86_verified.jsonl"
    "$WS/runs/harvest_gle87_verified.jsonl"
    "$WS/runs/harvest_gle92_verified.jsonl"
    "$WS/runs/harvest_gle93_verified.jsonl"
)
for f in "${need[@]}"; do
    if [[ ! -f "$f" ]] || [[ "$(wc -l < "$f")" -le 0 ]]; then
        echo "waiting: $f"
        exit 2
    fi
done

if ! python3 - "$FORGE/verify" "${need[@]}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from fast_verify import timing_complete
for path in sys.argv[2:]:
    if not timing_complete(path):
        raise SystemExit(1)
PY
then
    echo "waiting: GL-E harvest timing incomplete"
    exit 2
fi

if [[ -f "$MERGED/processor_config.json" ]]; then
    echo "already merged $MERGED"
    exit 0
fi

echo "=== GL-F speed gate ==="
gate_args=()
for lv in 86 87 92 93; do
    gate_args+=(--run "$lv:$WS/runs/harvest_gle${lv}_verified.jsonl")
done
if ! python3 "$FORGE/rl/glf_speed_gate.py" "${gate_args[@]}" \
        | tee "$WS/runs/glf_speed_gate.txt"; then
    echo "GATE failed; not training a speed SFT"
    exit 1
fi

echo "=== build best-of-N SFT ==="
if [[ -f "$WS/runs/sft_glf.jsonl" ]]; then
    echo "sft jsonl exists, skip build"
else
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=none NAME=glf_mix \
        "$FORGE/kernelbench/scripts/in_container.sh" \
        "cd /ws/cuTileForge && CUTILE_WS=/ws bash rl/build_glf_sft.sh /ws/runs/sft_glf.jsonl"
fi

if [[ ! -s "$WS/runs/sft_glf.jsonl" ]]; then
    echo "best-of-N jsonl empty; not training"
    exit 1
fi

echo "=== SFT + merge ==="
bash "$FORGE/rl/run_glf_sft.sh"
echo "pipeline done: $MERGED"
