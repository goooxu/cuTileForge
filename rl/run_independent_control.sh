#!/usr/bin/env bash
# Four independent same-prompt candidates for one model/task set.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
TAG="${1:?usage: run_independent_control.sh TAG MODEL LEVEL RUN_NAME [LIMIT]}"
MODEL="${2:?}"
LEVEL="${3:?}"
RUN_NAME="${4:?}"
LIMIT="${5:-0}"
RUN_DIR="$WS/runs/$RUN_NAME"
OUT="$WS/runs/${RUN_NAME}_verified.jsonl"
K=4

if [[ ! -d "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi

export CUTILE_WS="$WS"
# shellcheck source=sequential_model_env.sh
source "$FORGE/rl/sequential_model_env.sh"
configure_sequential_model "$TAG" "$MODEL"
export PROMPT_TIER=cutile_concepts
export TEMPERATURE=1.0
export NUM_WORKERS="${NUM_WORKERS:-32}"

n_level="$(python3 - "$KB/KernelBench/level$LEVEL" <<'PY'
import os, sys
print(sum(name.endswith(".py") for name in os.listdir(sys.argv[1])))
PY
)"
n_tasks="$n_level"
if [[ "$LIMIT" -gt 0 && "$LIMIT" -lt "$n_tasks" ]]; then
    n_tasks="$LIMIT"
fi
expected=$((n_tasks * K))

have() {
    python3 - "$RUN_DIR" <<'PY'
import os, sys
d = sys.argv[1]
print(sum(name.endswith("_kernel.py") for name in os.listdir(d))
      if os.path.isdir(d) else 0)
PY
}

timing_complete() {
    [[ -f "$OUT" ]] || return 1
    python3 - "$OUT" "$FORGE/verify" "$expected" <<'PY'
import json
import sys
sys.path.insert(0, sys.argv[2])
from worker import INCONCLUSIVE_STAGES
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
inconclusive = sum(
    row.get("stage") in INCONCLUSIVE_STAGES for row in rows)
cap = max(8, int(len(rows) * 0.01))
untimed = sum(
    bool(row.get("passed")) and not row.get("kernel_ms") for row in rows)
complete = (
    len(rows) >= int(sys.argv[3])
    and len(rows) - inconclusive > 0
    and inconclusive <= cap
    and untimed <= cap
)
raise SystemExit(0 if complete else 1)
PY
}

correctness_complete() {
    [[ -f "$OUT" ]] || return 1
    python3 - "$OUT" "$FORGE/verify" "$expected" <<'PY'
import json
import sys
sys.path.insert(0, sys.argv[2])
from worker import INCONCLUSIVE_STAGES
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
inconclusive = sum(
    row.get("stage") in INCONCLUSIVE_STAGES for row in rows)
cap = max(8, int(len(rows) * 0.01))
complete = (
    len(rows) >= int(sys.argv[3])
    and len(rows) - inconclusive > 0
    and inconclusive <= cap
)
raise SystemExit(0 if complete else 1)
PY
}

server_up() {
    curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1
}

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

wait_cuda() {
    for _ in $(seq 1 12); do
        if CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=all \
                "$KB/scripts/in_container.sh" \
                "python3 -c 'import torch; assert torch.cuda.device_count() >= 4'" \
                >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "CUDA did not return after stopping vLLM" >&2
    return 1
}

start_server() {
    for attempt in 1 2 3; do
        echo "[control] starting $TAG server, attempt $attempt"
        docker rm -f qwen-vllm >/dev/null 2>&1 || true
        "$KB/scripts/serve_qwen.sh" >/dev/null
        for _ in $(seq 1 180); do
            server_up && return 0
            if ! container_running qwen-vllm; then
                docker logs qwen-vllm 2>&1 | tail -30 || true
                break
            fi
            sleep 10
        done
    done
    return 1
}

if ! timing_complete; then
    for attempt in 1 2 3 4 5 6; do
        got="$(have)"
        [[ "$got" -ge "$expected" ]] && break
        server_up || start_server || exit 1
        GEN="ctrlgen_${RUN_NAME}"
        docker rm -f "$GEN" >/dev/null 2>&1 || true
        limit_arg=""
        if [[ "$LIMIT" -gt 0 ]]; then
            limit_arg="--limit $LIMIT"
        fi
        CUTILE_WS="$WS" GPUS=none DETACH=1 NAME="$GEN" \
            "$KB/scripts/in_container.sh" \
            "cd /ws/cuTileForge && python3 -u repair/sequential_search.py control \
                --run-dir /ws/runs/$RUN_NAME --level $LEVEL --samples $K \
                --prompt-tier cutile_concepts $limit_arg \
                --max-tokens $MAX_TOKENS --tokenizer $MODEL \
                --native-context $NATIVE_CONTEXT \
                --safety-margin ${SEQUENTIAL_SAFETY_MARGIN:-256} \
                --temperature 1.0 --top-p 0.95 --top-k 40 --concurrency 32" \
            >/dev/null
        while container_running "$GEN"; do
            sleep 20
        done
        docker logs "$GEN" 2>&1 | tail -8 || true
        docker rm "$GEN" >/dev/null 2>&1 || true
        after="$(have)"
        if [[ "$after" -le "$got" ]]; then
            docker rm -f qwen-vllm >/dev/null 2>&1 || true
        fi
    done
fi
if [[ "$(have)" -lt "$expected" ]]; then
    echo "control generation incomplete: $(have) / $expected" >&2
    exit 1
fi

docker rm -f qwen-vllm >/dev/null 2>&1 || true
wait_cuda

if ! correctness_complete; then
    FV="ctrlfv_${RUN_NAME}"
    docker rm -f "$FV" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="$FV" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$RUN_NAME --level $LEVEL \
            --workers 16 --gpus 4 --timeout 180 \
            --out /ws/runs/${RUN_NAME}_verified.jsonl"
    while container_running "$FV"; do
        sleep 20
    done
    docker logs "$FV" 2>&1 | tail -8 || true
    docker rm "$FV" >/dev/null 2>&1 || true
fi

if ! timing_complete; then
    TV="ctrltv_${RUN_NAME}"
    docker rm -f "$TV" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="$TV" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$RUN_NAME --level $LEVEL \
            --workers 4 --gpus 4 --timeout 180 --measure-time \
            --timing-from /ws/runs/${RUN_NAME}_verified.jsonl \
            --ref-mode compile --out /ws/runs/${RUN_NAME}_verified.jsonl"
    while container_running "$TV"; do
        sleep 20
    done
    docker logs "$TV" 2>&1 | tail -10 || true
    docker rm "$TV" >/dev/null 2>&1 || true
fi

timing_complete || {
    echo "control verification/timing incomplete" >&2
    exit 1
}
echo "independent control complete: $OUT"
