#!/usr/bin/env bash
# Four verifier-guided candidates for one model and one frozen task level.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
TAG="${1:?usage: run_sequential_search.sh TAG MODEL LEVEL RUN_NAME [LIMIT]}"
MODEL="${2:?}"
LEVEL="${3:?}"
RUN_NAME="${4:?}"
LIMIT="${5:-0}"
RUN_DIR="$WS/runs/$RUN_NAME"

if [[ ! -d "$MODEL" ]]; then
    echo "missing model: $MODEL" >&2
    exit 1
fi
if [[ ! -d "$KB/KernelBench/level$LEVEL" ]]; then
    echo "missing level $LEVEL" >&2
    exit 1
fi

export CUTILE_WS="$WS"
# shellcheck source=sequential_model_env.sh
source "$FORGE/rl/sequential_model_env.sh"
configure_sequential_model "$TAG" "$MODEL"

as_container() {
    local path="$1"
    case "$path" in
        "$WS"/*) echo "/ws/${path#"$WS"/}" ;;
        *) echo "$path" ;;
    esac
}

RUN_C="$(as_container "$RUN_DIR")"
init_args=(
    "cd /ws/cuTileForge && python3 repair/sequential_search.py init"
    "--run-dir $RUN_C --tag $TAG --level $LEVEL --prompt-tier cutile_concepts"
)
if [[ "$LIMIT" -gt 0 ]]; then
    init_args+=("--limit $LIMIT")
fi
CUTILE_WS="$WS" GPUS=none "$KB/scripts/in_container.sh" \
    "${init_args[*]}"

n_tasks="$(python3 - "$RUN_DIR/state.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["tasks"]))
PY
)"

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
    local attempt
    for attempt in 1 2 3; do
        echo "[sequential] starting $TAG server, attempt $attempt"
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

round_have() {
    local round="$1"
    python3 - "$RUN_DIR/round_$round" <<'PY'
import os, sys
d = sys.argv[1]
print(sum(name.endswith("_kernel.py") for name in os.listdir(d))
      if os.path.isdir(d) else 0)
PY
}

timing_complete() {
    local path="$1"
    [[ -f "$path" ]] || return 1
    python3 - "$path" "$FORGE/verify" "$n_tasks" <<'PY'
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
    local path="$1"
    [[ -f "$path" ]] || return 1
    python3 - "$path" "$FORGE/verify" "$n_tasks" <<'PY'
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

for round in 0 1 2 3; do
    round_dir="$RUN_DIR/round_$round"
    verified="$RUN_DIR/round_${round}_verified.jsonl"
    if [[ -f "$round_dir/updated.json" ]] && timing_complete "$verified"; then
        echo "=== $TAG $RUN_NAME round $round already complete ==="
        continue
    fi

    echo "=== $TAG $RUN_NAME round $round generate ==="
    for attempt in 1 2 3 4 5 6; do
        have="$(round_have "$round")"
        [[ "$have" -ge "$n_tasks" ]] && break
        server_up || start_server || {
            echo "could not start $TAG server" >&2
            exit 1
        }
        GEN="seqgen_${RUN_NAME}_${round}"
        docker rm -f "$GEN" >/dev/null 2>&1 || true
        CUTILE_WS="$WS" GPUS=none DETACH=1 NAME="$GEN" \
            "$KB/scripts/in_container.sh" \
            "cd /ws/cuTileForge && python3 -u repair/sequential_search.py generate \
                --run-dir $RUN_C --round $round --max-tokens $MAX_TOKENS \
                --tokenizer $MODEL --native-context $NATIVE_CONTEXT \
                --safety-margin ${SEQUENTIAL_SAFETY_MARGIN:-256} \
                --temperature 1.0 --top-p 0.95 --top-k 40 --concurrency 32"
        while container_running "$GEN"; do
            sleep 20
        done
        docker logs "$GEN" 2>&1 | tail -8 || true
        docker rm "$GEN" >/dev/null 2>&1 || true
        after="$(round_have "$round")"
        if [[ "$after" -le "$have" ]]; then
            echo "round $round made no progress; restart server"
            docker rm -f qwen-vllm >/dev/null 2>&1 || true
        fi
    done
    if [[ "$(round_have "$round")" -lt "$n_tasks" ]]; then
        echo "round $round generation incomplete" >&2
        exit 1
    fi

    echo "=== stop $TAG server before verification ==="
    docker rm -f qwen-vllm >/dev/null 2>&1 || true
    wait_cuda

    if ! correctness_complete "$verified"; then
        echo "=== $TAG $RUN_NAME round $round correctness ==="
        FV="seqfv_${RUN_NAME}_${round}"
        docker rm -f "$FV" >/dev/null 2>&1 || true
        CUTILE_WS="$WS" DETACH=1 NAME="$FV" "$KB/scripts/in_container.sh" \
            "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
                --kernel-dir $RUN_C/round_$round --level $LEVEL \
                --workers 16 --gpus 4 --timeout 180 \
                --out $RUN_C/round_${round}_verified.jsonl"
        while container_running "$FV"; do
            sleep 20
        done
        docker logs "$FV" 2>&1 | tail -8 || true
        docker rm "$FV" >/dev/null 2>&1 || true
    fi

    if ! timing_complete "$verified"; then
        echo "=== $TAG $RUN_NAME round $round kernel_ms ==="
        TV="seqtv_${RUN_NAME}_${round}"
        docker rm -f "$TV" >/dev/null 2>&1 || true
        CUTILE_WS="$WS" DETACH=1 NAME="$TV" "$KB/scripts/in_container.sh" \
            "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
                --kernel-dir $RUN_C/round_$round --level $LEVEL \
                --workers 4 --gpus 4 --timeout 180 --measure-time \
                --timing-from $RUN_C/round_${round}_verified.jsonl \
                --ref-mode compile --out $RUN_C/round_${round}_verified.jsonl"
        while container_running "$TV"; do
            sleep 20
        done
        docker logs "$TV" 2>&1 | tail -10 || true
        docker rm "$TV" >/dev/null 2>&1 || true
    fi
    if ! timing_complete "$verified"; then
        echo "round $round verification/timing incomplete" >&2
        exit 1
    fi

    CUTILE_WS="$WS" GPUS=none "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 repair/sequential_search.py update \
            --run-dir $RUN_C --round $round \
            --verified $RUN_C/round_${round}_verified.jsonl"
done

CUTILE_WS="$WS" GPUS=none "$KB/scripts/in_container.sh" \
    "cd /ws/cuTileForge && python3 repair/sequential_search.py status \
        --run-dir $RUN_C"
echo "sequential search complete: $RUN_DIR"
