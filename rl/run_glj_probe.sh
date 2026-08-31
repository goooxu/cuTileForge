#!/usr/bin/env bash
# Run one merged Glimmer checkpoint on the frozen GL-J scale-selection probe.
#
# This is not table A.  Level 62 contains 32 matmul + 32 conv tasks copied from
# training levels after excluding every GL-I ORPO/retain task.  Sampling matches
# table A so the probe measures the same generation behavior, but it never
# touches levels 60/84/88/97/98/99.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
TAG="${1:?usage: run_glj_probe.sh <tag> <model>}"
MODEL="${2:?usage: run_glj_probe.sh <tag> <model>}"
LEVEL=62
K=4
RUN_NAME="${TAG}_glj_probe"
RUN_DIR="$WS/runs/$RUN_NAME"
OUT="$WS/runs/${RUN_NAME}_verified.jsonl"
EXPECTED=256

if [[ ! -d "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi
if [[ ! -f "$WS/runs/glj_probe_manifest.json" ]]; then
    echo "missing glj_probe_manifest.json; run rl/build_glj_probe.py" >&2
    exit 1
fi
n_tasks="$(python3 - "$KB/KernelBench/level$LEVEL" <<'PY'
import os, sys
print(sum(name.endswith(".py") for name in os.listdir(sys.argv[1])))
PY
)"
if [[ "$n_tasks" -ne 64 ]]; then
    echo "level $LEVEL is not the frozen 64-problem probe" >&2
    exit 1
fi

export CUTILE_WS="$WS"
export MODEL
export MOUNTS="-v /raid/tmp:/raid/tmp:ro"
export VLLM_IMAGE=vllm/vllm-openai:muse-glimmer
export REASONING_STRENGTH=xhigh
export KEEP_SPECIAL_TOKENS=1
export EXTRA_ARGS="--reasoning-parser muse_glimmer --generation-config auto"
export MAX_TOKENS=32768
export PROMPT_TIER=cutile_concepts
export TEMPERATURE=1.0
export NUM_WORKERS="${NUM_WORKERS:-32}"

timing_complete() {
    [[ -f "$OUT" ]] || return 1
    python3 - "$OUT" "$FORGE/verify" "$EXPECTED" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from fast_verify import timing_complete
raise SystemExit(0 if timing_complete(sys.argv[1], need=int(sys.argv[3])) else 1)
PY
}

if timing_complete; then
    echo "$TAG probe already complete: $OUT"
    exit 0
fi

echo "=== $TAG probe: generate level $LEVEL k=$K ==="
bash "$FORGE/taskgen/generate_with_restart.sh" "$RUN_NAME" "$LEVEL" "$K" 8

echo "=== stop vLLM ==="
docker rm -f qwen-vllm >/dev/null 2>&1 || true
sleep 5

have=0
[[ -f "$OUT" ]] && have="$(wc -l < "$OUT")"
if [[ "$have" -ne "$EXPECTED" ]]; then
    echo "=== $TAG probe: correctness ==="
    FV="fv_${RUN_NAME}"
    docker rm -f "$FV" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="$FV" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$RUN_NAME --level $LEVEL \
            --workers 16 --gpus 4 --timeout 180 \
            --out /ws/runs/${RUN_NAME}_verified.jsonl"
    while docker ps --filter name="$FV" --format '{{.Names}}' | grep -q "$FV"; do
        sleep 30
    done
    docker logs "$FV" 2>&1 | tail -8 || true
    docker rm "$FV" >/dev/null 2>&1 || true
fi

echo "=== $TAG probe: kernel_ms timing ==="
TV="tv_${RUN_NAME}"
docker rm -f "$TV" >/dev/null 2>&1 || true
CUTILE_WS="$WS" DETACH=1 NAME="$TV" "$KB/scripts/in_container.sh" \
    "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
        --kernel-dir /ws/runs/$RUN_NAME --level $LEVEL \
        --workers 4 --gpus 4 --timeout 180 \
        --measure-time --timing-from /ws/runs/${RUN_NAME}_verified.jsonl \
        --ref-mode compile --out /ws/runs/${RUN_NAME}_verified.jsonl"
while docker ps --filter name="$TV" --format '{{.Names}}' | grep -q "$TV"; do
    sleep 30
done
docker logs "$TV" 2>&1 | tail -10 || true
docker rm "$TV" >/dev/null 2>&1 || true

if ! timing_complete; then
    echo "$TAG probe timing incomplete" >&2
    exit 1
fi
echo "$TAG probe complete: $OUT"
