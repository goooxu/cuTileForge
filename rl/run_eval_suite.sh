#!/usr/bin/env bash
# Sample a model on the standalone eval suite (level 60 + 61) and score it.
#
# Frozen protocol: cutile_concepts, TILE=1024, temperature 1.0, k=4.
# Correctness is not timed. Speed is timed against torch.compile.
#
# Usage:
#   MODEL=/raid/tmp/gemsg-cutile/model-M ./rl/run_eval_suite.sh M
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

TAG="${1:?usage: run_eval_suite.sh <tag> [--smoke]}"
shift || true
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
    shift || true
fi

K="${EVAL_K:-4}"
if [[ "$SMOKE" -eq 1 ]]; then
    echo "SMOKE: first 16 problems per track; not a published score"
    SUBSET_START=1
    SUBSET_END=16
else
    SUBSET_START=1
    SUBSET_END=0
fi

bash "$FORGE/overlay/scripts/install_eval_suite.sh"

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
    local n expected
    n="$(n_problems "$level")"
    expected=$((n * K))
    echo "=== generate $run_name k=$K expected=$expected ==="
    for round in $(seq 1 12); do
        got="$(have "$run_dir")"
        echo "=== $run_name round $round: have $got / $expected ==="
        [[ "$got" -ge "$expected" ]] && return 0
        server_up || start_server || return 1
        extra=()
        if [[ "$SUBSET_END" -gt 0 ]]; then
            extra+=("subset=($SUBSET_START, $SUBSET_END)")
        fi
        NAME="gen-$run_name" "$KB/scripts/run_generate.sh" "$run_name" "$level" "$K" \
            "${extra[@]}" log_raw_response=True 2>&1 | tail -5
        docker rm -f "gen-$run_name" >/dev/null 2>&1 || true
    done
    got="$(have "$run_dir")"
    [[ "$got" -ge "$expected" ]]
}

verify_level() {
    local level="$1"
    local timed="$2"
    local run_name="${TAG}_l${level}"
    local out="$WS/runs/${run_name}_verified.jsonl"
    local extra=""
    if [[ "$timed" == "1" ]]; then
        extra="--measure-time --ref-mode compile --timeout 180"
    else
        extra="--timeout 120"
    fi
    echo "=== verify $run_name timed=$timed ==="
    docker rm -f "fv_$run_name" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="fv_$run_name" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$run_name --level $level \
            --workers 16 --gpus 4 --out /ws/runs/${run_name}_verified.jsonl \
            $extra"
    while docker ps --filter name="fv_$run_name" --format '{{.Names}}' \
            | grep -q "fv_$run_name"; do
        sleep 30
    done
    echo "verify done: $out"
}

generate_level 60 || { echo "generation failed on level 60"; exit 1; }
generate_level 61 || { echo "generation failed on level 61"; exit 1; }

echo "=== stop vLLM ==="
docker rm -f qwen-vllm >/dev/null 2>&1 || true

verify_level 60 0
verify_level 61 1

python3 "$FORGE/verify/eval_scorecard.py" --run "$TAG:$WS/runs/${TAG}" --k "$K"
