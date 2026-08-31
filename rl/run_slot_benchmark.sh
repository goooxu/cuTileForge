#!/usr/bin/env bash
# Measure 32/64/96/128 work-conserving request slots on two generation GPUs.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
LOCK="$WS/runs/.slot_benchmark.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another slot benchmark holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.slot_benchmark.pid"

bash "$FORGE/rl/prepare_sequential_models.sh"
# shellcheck disable=SC1090
source "$WS/runs/sequential_models.env"
bash "$FORGE/overlay/scripts/install_eval_suite.sh"

export SEQUENTIAL_USE_NATIVE_CONTEXT=1
export SEQUENTIAL_OUTPUT_CAP=131072
export SEQUENTIAL_SAFETY_MARGIN=1024

tags=(GLE GL Q base G4t Q38)
if [[ -n "${SLOT_TAGS:-}" ]]; then
    read -r -a tags <<< "$SLOT_TAGS"
fi
SLOT_SUFFIX="${SLOT_SUFFIX:-}"
SLOT_CONCURRENCIES="${SLOT_CONCURRENCIES:-32,64,96,128}"
SLOT_WARMUP="${SLOT_WARMUP:-15}"
SLOT_TASKS="${SLOT_TASKS:-8}"
SLOT_DURATION="${SLOT_DURATION:-60}"

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

server_up() {
    curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1
}

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    case "$tag" in
        GLE) model="$GLE_MODEL" ;;
        GL) model="$GL_MODEL" ;;
        Q) model="$Q_MODEL" ;;
        base) model="$BASE_MODEL" ;;
        G4t) model="$G4T_MODEL" ;;
        Q38) model="$Q38_MODEL" ;;
        *) echo "unknown slot benchmark tag: $tag" >&2; exit 1 ;;
    esac
    out="$WS/runs/slot_bench_${SLOT_SUFFIX}${tag}.json"
    log="$WS/runs/slot_bench_${SLOT_SUFFIX}${tag}.log"
    if [[ -s "$out" ]]; then
        echo "$tag slot benchmark already complete"
        continue
    fi

    # shellcheck source=sequential_model_env.sh
    source "$FORGE/rl/sequential_model_env.sh"
    configure_sequential_model "$tag" "$model"
    export VLLM_GPUS=device=0,1
    export TENSOR_PARALLEL=2
    export GPU_UTIL=0.90

    ready=0
    for attempt in 1 2 3; do
        echo "=== $tag TP2 slot benchmark server attempt $attempt ==="
        docker rm -f qwen-vllm >/dev/null 2>&1 || true
        "$KB/scripts/serve_qwen.sh" >/dev/null
        for _ in $(seq 1 180); do
            if server_up; then
                ready=1
                break
            fi
            if ! container_running qwen-vllm; then
                docker logs --tail 25 qwen-vllm 2>&1 || true
                break
            fi
            sleep 10
        done
        [[ "$ready" == "1" ]] && break
    done
    if [[ "$ready" != "1" ]]; then
        echo "$tag TP2 server failed" >&2
        exit 1
    fi

    docker rm -f "slotbench_$tag" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest \
        GPUS=device=0,1 NAME="slotbench_$tag" \
        "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u repair/benchmark_slots.py \
            --tag $tag --tokenizer $model --native-context $NATIVE_CONTEXT \
            --max-tokens 131072 --safety-margin 1024 \
            --level 60 --tasks $SLOT_TASKS \
            --concurrencies $SLOT_CONCURRENCIES \
            --warmup $SLOT_WARMUP --duration $SLOT_DURATION \
            --out /ws/runs/slot_bench_${SLOT_SUFFIX}${tag}.json" \
        | tee "$log"
    docker rm -f qwen-vllm >/dev/null 2>&1 || true
done

echo "six-model slot benchmark complete"
