#!/usr/bin/env bash
# Sample a model on the standalone eval suite (level 60) and score it.
#
# Frozen protocol: cutile_concepts, TILE=1024, temperature 1.0, k=4,
# max_tokens=32768. Qwen3.8 thinking: ENABLE_THINKING=1/0 forces on/off
# (the model default is on; unset leaves that default).
# Every problem is timed against torch.compile. The scorecard splits
# latency (770) from throughput twins.
#
# Usage:
#   MODEL=/path/to/model-M ./rl/run_eval_suite.sh M
#   MODEL=... ./rl/run_eval_suite.sh M --smoke
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
KB="$FORGE/kernelbench"

export CUTILE_WS="$WS"
export MODEL="${MODEL:?MODEL must point at the merged checkpoint to sample from}"
export MOUNTS="${MOUNTS:--v /raid/tmp:/raid/tmp:ro}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
export PROMPT_TIER="${PROMPT_TIER:-cutile_concepts}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export NUM_WORKERS="${NUM_WORKERS:-32}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export ENABLE_THINKING="${ENABLE_THINKING:-}"
export REASONING_EFFORT="${REASONING_EFFORT:-}"
export REASONING_STRENGTH="${REASONING_STRENGTH:-}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"
export TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"

TAG="${1:?usage: run_eval_suite.sh <tag> [--smoke]}"
shift || true
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
    shift || true
fi

K="${EVAL_K:-4}"
if [[ "$SMOKE" -eq 1 ]]; then
    echo "SMOKE: first 16 problems; not a published score"
    SUBSET_START=1
    SUBSET_END=16
else
    SUBSET_START=1
    SUBSET_END=0
fi

# SKIP_INSTALL=1 when another host is already verifying from this
# checkout: install_eval_suite.sh wipes KernelBench/level60 first.
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    bash "$FORGE/overlay/scripts/install_eval_suite.sh"
fi

have() {
    python3 - "$1" <<'PY'
import os, sys
d = sys.argv[1]
print(sum(1 for f in os.listdir(d) if f.endswith("_kernel.py")) if os.path.isdir(d) else 0)
PY
}

n_problems() {
    python3 - "$KB/KernelBench/level$1" "$SUBSET_END" <<'PY'
import os, sys
d, end = sys.argv[1], int(sys.argv[2])
n = sum(1 for f in os.listdir(d) if f.endswith(".py"))
print(n if end <= 0 else min(n, end))
PY
}

expected_n() {
    python3 - "$FORGE/tasks/eval/manifest.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["problems"]))
PY
}

server_up() { curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1; }

start_server() {
    echo "[evalsuite] (re)starting vLLM"
    docker rm -f qwen-vllm >/dev/null 2>&1 || true
    "$KB/scripts/serve_qwen.sh" >/dev/null 2>&1
    for _ in $(seq 1 180); do
        server_up && { echo "[evalsuite] server up"; return 0; }
        docker ps --filter name=qwen-vllm --format '{{.Names}}' | grep -q qwen-vllm || {
            echo "[evalsuite] container died during startup"; return 1; }
        sleep 10
    done
    echo "[evalsuite] server did not come up in time"
    return 1
}

generate_level() {
    local level="$1"
    local run_name="${TAG}_l${level}"
    local run_dir="$WS/runs/$run_name"
    local n want
    n="$(n_problems "$level")"
    if [[ "$SMOKE" -eq 0 ]]; then
        want="$(expected_n)"
        if [[ "$n" -ne "$want" ]]; then
            echo "level $level has $n problems, expected $want; install failed" >&2
            return 1
        fi
    fi
    local expected=$((n * K))
    echo "=== generate $run_name k=$K expected=$expected thinking=${ENABLE_THINKING:-unset} max_tokens=$MAX_TOKENS ==="
    # First generation round always (re)starts vLLM. compare switches
    # MODEL between tags, but the served name is always Qwen3-Coder-Next,
    # so a leftover server is the wrong weights.
    fresh_server=1
    for round in $(seq 1 12); do
        got="$(have "$run_dir")"
        echo "=== $run_name round $round: have $got / $expected ==="
        [[ "$got" -ge "$expected" ]] && return 0
        if [[ "$fresh_server" -eq 1 ]]; then
            start_server || return 1
            fresh_server=0
        else
            server_up || start_server || return 1
        fi
        extra=()
        if [[ "$SUBSET_END" -gt 0 ]]; then
            extra+=("subset=($SUBSET_START, $SUBSET_END)")
        fi
        # Detached: a foreground docker run --rm dies with this bash tree
        # (that is how the Q38 6h generate stopped at 22:59 with no OOM).
        NAME="gen-$run_name" DETACH=1 "$KB/scripts/run_generate.sh" \
            "$run_name" "$level" "$K" \
            "${extra[@]}" log_raw_response=True >/dev/null
        while docker ps --filter name="gen-$run_name" --format '{{.Names}}' \
                | grep -q "gen-$run_name"; do
            sleep 30
        done
        docker logs "gen-$run_name" 2>&1 | tail -5 || true
        docker rm -f "gen-$run_name" >/dev/null 2>&1 || true
    done
    got="$(have "$run_dir")"
    [[ "$got" -ge "$expected" ]]
}

verify_level() {
    local level="$1"
    local run_name="${TAG}_l${level}"
    local out="$WS/runs/${run_name}_verified.jsonl"
    # Two containers: after the screening pool exits, CUDA in that container
    # can fail to re-init (G4t: timing workers hit "No CUDA GPUs are available"
    # and the parent wedged). Timing always starts clean.
    echo "=== verify $run_name correctness ==="
    docker rm -f "fv_$run_name" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="fv_$run_name" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$run_name --level $level \
            --workers 16 --gpus 4 --out /ws/runs/${run_name}_verified.jsonl \
            --timeout 180"
    while docker ps --filter name="fv_$run_name" --format '{{.Names}}' \
            | grep -q "fv_$run_name"; do
        sleep 30
    done
    echo "=== verify $run_name timing (fresh container) ==="
    docker rm -f "fv_$run_name" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="fv_$run_name" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$run_name --level $level \
            --workers 4 --gpus 4 --out /ws/runs/${run_name}_verified.jsonl \
            --measure-time --timing-from /ws/runs/${run_name}_verified.jsonl \
            --ref-mode compile --timeout 180"
    while docker ps --filter name="fv_$run_name" --format '{{.Names}}' \
            | grep -q "fv_$run_name"; do
        sleep 30
    done
    echo "verify done: $out"
}

generate_level 60 || { echo "generation failed on level 60"; exit 1; }

echo "=== stop vLLM ==="
docker rm -f qwen-vllm >/dev/null 2>&1 || true

verify_level 60

python3 "$FORGE/verify/eval_scorecard.py" --run "$TAG:$WS/runs/${TAG}" --k "$K" \
    || echo "scorecard failed for $TAG (non-fatal; compare continues)"
