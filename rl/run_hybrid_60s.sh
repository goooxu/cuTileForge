#!/usr/bin/env bash
# One-model 60-second Hybrid search plus formal validation and scoring.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
TAG="${1:?usage: run_hybrid_60s.sh TAG MODEL LEVEL RUN_NAME [LIMIT]}"
MODEL_PATH="${2:?}"
LEVEL="${3:?}"
RUN_NAME="${4:?}"
LIMIT="${5:-0}"
RUN_DIR="$WS/runs/$RUN_NAME"
RUN_C="/ws/runs/$RUN_NAME"
SEARCH_CONTAINER="hybrid_${RUN_NAME//[^a-zA-Z0-9_.-]/_}"
HYBRID_PROFILE="${HYBRID_PROFILE:-60s}"
HYBRID_DEADLINE="${HYBRID_DEADLINE:-60}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "missing model: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -d "$KB/KernelBench/level$LEVEL" ]]; then
    echo "missing level $LEVEL" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
if [[ -s "$RUN_DIR/complete.json" ]]; then
    if python3 - "$RUN_DIR/state.json" "$RUN_DIR/complete.json" \
            "$HYBRID_PROFILE" <<'PY'
import json
import sys
state = json.load(open(sys.argv[1]))
complete = json.load(open(sys.argv[2]))
terminal = {"solved", "timeout", "stagnated"}
done = all(
    row.get("status") in terminal for row in state["tasks"].values())
if sys.argv[3] == "600s":
    validated = complete.get("official_validated", 0)
    cap = max(8, validated // 100)
    done = done and (
        complete.get("official_timed", 0) >= validated - cap)
raise SystemExit(0 if done else 1)
PY
    then
        echo "Hybrid search already complete: $RUN_DIR"
        exit 0
    fi
fi

export CUTILE_WS="$WS"
export SEQUENTIAL_USE_NATIVE_CONTEXT=1
export SEQUENTIAL_OUTPUT_CAP=131072
export SEQUENTIAL_SAFETY_MARGIN=1024
# shellcheck source=sequential_model_env.sh
source "$FORGE/rl/sequential_model_env.sh"
configure_sequential_model "$TAG" "$MODEL_PATH"
export VLLM_GPUS=device=0,1
export TENSOR_PARALLEL=2
export GPU_UTIL="${HYBRID_GPU_UTIL:-0.90}"

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

server_up() {
    curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1
}

stop_server() {
    docker rm -f qwen-vllm >/dev/null 2>&1 || true
}

capture_search_log() {
    if docker inspect "$SEARCH_CONTAINER" >/dev/null 2>&1; then
        docker logs "$SEARCH_CONTAINER" >"$RUN_DIR/search.log" 2>&1 || true
    fi
}

cleanup() {
    capture_search_log
    docker rm -f "$SEARCH_CONTAINER" >/dev/null 2>&1 || true
    stop_server
}
trap cleanup EXIT

start_server() {
    local attempt
    for attempt in 1 2 3; do
        echo "[hybrid] starting $TAG TP2 server, attempt $attempt"
        stop_server
        "$KB/scripts/serve_qwen.sh" >/dev/null
        for _ in $(seq 1 210); do
            if server_up; then
                return 0
            fi
            if ! container_running qwen-vllm; then
                docker logs --tail 40 qwen-vllm 2>&1 || true
                break
            fi
            sleep 10
        done
    done
    return 1
}

wait_cuda() {
    for _ in $(seq 1 24); do
        if CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=all \
                "$KB/scripts/in_container.sh" \
                "python3 -c 'import torch; assert torch.cuda.device_count() >= 4; torch.cuda.init()'" \
                >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "CUDA did not return after stopping Hybrid containers" >&2
    return 1
}

search_complete() {
    [[ -s "$RUN_DIR/state.json" ]] || return 1
    python3 - "$RUN_DIR/state.json" <<'PY'
import json
import sys
state = json.load(open(sys.argv[1]))
terminal = {"solved", "timeout", "stagnated"}
raise SystemExit(0 if all(
    row.get("status") in terminal for row in state["tasks"].values()) else 1)
PY
}

args=(
    "cd /ws/cuTileForge && python3 -u repair/hybrid_search.py run"
    "--run-dir $RUN_C --tag $TAG --level $LEVEL"
    "--tokenizer $MODEL_PATH --native-context $NATIVE_CONTEXT"
    "--max-tokens 131072 --safety-margin 1024"
    "--profile $HYBRID_PROFILE --deadline $HYBRID_DEADLINE"
    "--request-timeout $((HYBRID_DEADLINE + 60))"
    "--temperature 1.0 --top-p 0.95 --top-k 40"
    "--verify-workers 4 --verify-gpus 2"
    "--verify-timeout ${HYBRID_VERIFY_TIMEOUT:-45}"
    "--verify-batch ${HYBRID_VERIFY_BATCH:-4}"
    "--backpressure-high ${HYBRID_BACKPRESSURE_HIGH:-64}"
    "--backpressure-low ${HYBRID_BACKPRESSURE_LOW:-32}"
)
if [[ "$LIMIT" -gt 0 ]]; then
    args+=("--limit $LIMIT")
fi
[[ -n "${HYBRID_GLOBAL_SLOTS:-}" ]] \
    && args+=("--global-slots $HYBRID_GLOBAL_SLOTS")
[[ -n "${HYBRID_ACTIVE_PROBLEMS:-}" ]] \
    && args+=("--active-problems $HYBRID_ACTIVE_PROBLEMS")
[[ -n "${HYBRID_PER_TASK_SLOTS:-}" ]] \
    && args+=("--per-task-slots $HYBRID_PER_TASK_SLOTS")

if search_complete; then
    echo "Hybrid search state already terminal: $RUN_DIR"
else
    start_server || {
        echo "could not start $TAG TP2 server" >&2
        exit 1
    }
    docker rm -f "$SEARCH_CONTAINER" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=device=2,3 \
        MOUNTS="$MOUNTS" DETACH=1 NAME="$SEARCH_CONTAINER" \
        "$KB/scripts/in_container.sh" "${args[*]}" >/dev/null

    while container_running "$SEARCH_CONTAINER"; do
        sleep 15
    done
    capture_search_log
    search_exit="$(docker inspect -f '{{.State.ExitCode}}' "$SEARCH_CONTAINER")"
    docker logs --tail 30 "$SEARCH_CONTAINER" 2>&1 || true
    docker rm "$SEARCH_CONTAINER" >/dev/null 2>&1 || true
    if [[ "$search_exit" != "0" ]]; then
        echo "$TAG Hybrid search failed with exit $search_exit" >&2
        exit "$search_exit"
    fi
fi

stop_server
wait_cuda

n_solved="$(python3 - "$RUN_DIR/state.json" <<'PY'
import json
import sys
state = json.load(open(sys.argv[1]))
print(sum(row.get("status") == "solved" for row in state["tasks"].values()))
PY
)"
post="$RUN_DIR/post_verified.jsonl"

if [[ "$n_solved" -eq 0 ]]; then
    : > "$post"
elif ! python3 - "$post" "$n_solved" <<'PY'
import json
import os
import sys
if not os.path.isfile(sys.argv[1]):
    raise SystemExit(1)
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
raise SystemExit(0 if len(rows) >= int(sys.argv[2]) else 1)
PY
then
    echo "=== $TAG official correctness validation: $n_solved frozen kernels ==="
    FV="hybrid_post_fv_${RUN_NAME//[^a-zA-Z0-9_.-]/_}"
    docker rm -f "$FV" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=all \
        DETACH=1 NAME="$FV" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir $RUN_C/frozen --level $LEVEL \
            --workers 16 --gpus 4 --timeout 180 \
            --out $RUN_C/post_verified.jsonl" >/dev/null
    while container_running "$FV"; do
        sleep 15
    done
    docker logs --tail 20 "$FV" 2>&1 || true
    fv_exit="$(docker inspect -f '{{.State.ExitCode}}' "$FV")"
    docker rm "$FV" >/dev/null 2>&1 || true
    [[ "$fv_exit" == "0" ]] || exit "$fv_exit"
fi

n_validated="$(python3 - "$post" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(sum(bool(row.get("passed")) for row in rows))
PY
)"
if [[ "$n_validated" -gt 0 ]]; then
    # A correctness pool can leave CUDA contexts tearing down after its
    # container exits. Starting timing immediately has produced workers with
    # "No CUDA GPUs are available"; wait for a clean all-GPU initialization.
    sleep 30
    wait_cuda
    echo "=== $TAG official kernel_ms scoring: $n_validated validated kernels ==="
    TV="hybrid_post_tv_${RUN_NAME//[^a-zA-Z0-9_.-]/_}"
    docker rm -f "$TV" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=all \
        DETACH=1 NAME="$TV" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir $RUN_C/frozen --level $LEVEL \
            --workers 4 --gpus 4 --timeout 180 --measure-time \
            --timing-from $RUN_C/post_verified.jsonl \
            --ref-mode compile --out $RUN_C/post_verified.jsonl" >/dev/null
    while container_running "$TV"; do
        sleep 15
    done
    docker logs --tail 20 "$TV" 2>&1 || true
    tv_exit="$(docker inspect -f '{{.State.ExitCode}}' "$TV")"
    docker rm "$TV" >/dev/null 2>&1 || true
    [[ "$tv_exit" == "0" ]] || exit "$tv_exit"
    n_timed="$(python3 - "$post" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(sum(
    bool(row.get("passed")) and row.get("kernel_ms") is not None
    for row in rows))
PY
)"
    timing_cap=$((n_validated / 100))
    [[ "$timing_cap" -lt 8 ]] && timing_cap=8
    if [[ "$n_timed" -lt $((n_validated - timing_cap)) ]]; then
        echo "official timing incomplete: $n_timed/$n_validated" >&2
        exit 1
    fi
fi

python3 - "$RUN_DIR/state.json" "$post" "$RUN_DIR/complete.json" <<'PY'
import json
import os
import sys
import time

state = json.load(open(sys.argv[1]))
terminal = {"solved", "timeout", "stagnated"}
bad = [
    pid for pid, row in state["tasks"].items()
    if row.get("status") not in terminal
]
if bad:
    raise SystemExit("nonterminal Hybrid tasks remain: %s" % bad[:8])
post = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
value = {
    "tag": state["tag"],
    "level": state["level"],
    "tasks": len(state["tasks"]),
    "solved_online": sum(
        row.get("status") == "solved" for row in state["tasks"].values()),
    "official_validated": sum(bool(row.get("passed")) for row in post),
    "official_timed": sum(
        bool(row.get("passed")) and row.get("kernel_ms") is not None
        for row in post),
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
tmp = sys.argv[3] + ".tmp"
with open(tmp, "w") as out:
    json.dump(value, out, indent=2, sort_keys=True)
    out.write("\n")
os.replace(tmp, sys.argv[3])
PY

trap - EXIT
cleanup
echo "Hybrid search complete: $RUN_DIR"
