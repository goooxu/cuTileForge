#!/usr/bin/env bash
# Screen GL-C learnable frontiers for the given levels, nothing else.
#
# Splits the 86/87/92/93 screen across two GPU boxes: the box already
# running run_gl_grpo.sh keeps going, this fills a second box that would
# otherwise sit idle. Writes rl_frontier_glc_<level>.json; the GRPO wrapper
# skips a level whose file already exists. Does not take the GRPO lock and
# does not start training.
#
# Usage:
#   CUTILE_WS=... LEVELS=92,93 rl/screen_gl_frontier.sh
set -uo pipefail

if [[ "${DAEMONIZE:-0}" = 1 ]]; then
    unset DAEMONIZE
    exec python3 - "$0" "$@" <<'PY'
import os, sys
script = sys.argv[1]
rest = sys.argv[2:]
if os.fork():
    sys.exit(0)
os.setsid()
if os.fork():
    sys.exit(0)
log = os.environ.get("SCREEN_LOG")
if log:
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
devnull = os.open("/dev/null", os.O_RDONLY)
os.dup2(devnull, 0)
os.close(devnull)
os.execve("/bin/bash", ["bash", script] + rest, os.environ)
PY
fi

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
# Always land stdout in a log when we were daemonized, so a closed SSH
# does not swallow "serving" / "screening" lines.
if [[ ! -t 1 ]]; then
    _log="${SCREEN_LOG:-$WS/runs/glc_front_screen.log}"
    mkdir -p "$(dirname "$_log")"
    exec >> "$_log" 2>&1
fi
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
SCRATCH="${RL_SCRATCH:-/raid/tmp/gemsg-cutile}"
MODEL="${MODEL:-$SCRATCH/model-GLC}"
LEVELS="${LEVELS:?LEVELS=92,93}"
SAMPLES="${SAMPLES:-6}"
HELD_OUT=" 60 84 88 97 98 99 "

export CUTILE_WS="$WS"
export MODEL
export MOUNTS="${MOUNTS:--v $SCRATCH:$SCRATCH}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:muse-glimmer}"
export EXTRA_ARGS="${EXTRA_ARGS:---generation-config auto}"
export REASONING_STRENGTH="${REASONING_STRENGTH:-xhigh}"
export KEEP_SPECIAL_TOKENS=1
export GPU_UTIL="${GPU_UTIL:-0.45}"
export PROMPT_TIER="${PROMPT_TIER:-cutile_concepts}"

if ! [[ "$VLLM_IMAGE" == *muse-glimmer* ]]; then
    echo "refusing VLLM_IMAGE=$VLLM_IMAGE (need muse-glimmer)" >&2
    exit 1
fi
if ! { [ -f "$MODEL/processor_config.json" ] || [ -f "$MODEL/config.json" ]; }; then
    echo "missing $MODEL" >&2
    exit 1
fi

serve_glc() {
    if curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
            | grep -q Qwen3; then
        echo "already serving"
        return 0
    fi
    docker rm -f qwen-vllm >/dev/null 2>&1 || true
    CUTILE_WS="$WS" MODEL="$MODEL" MOUNTS="-v $SCRATCH:$SCRATCH:ro" \
        GPU_UTIL="$GPU_UTIL" VLLM_IMAGE="$VLLM_IMAGE" EXTRA_ARGS="$EXTRA_ARGS" \
        "$KB/scripts/serve_qwen.sh"
    for _ in $(seq 1 60); do
        if curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
                | grep -q Qwen3; then
            echo "serving $MODEL"
            return 0
        fi
        sleep 15
    done
    echo "vLLM failed to come up on $MODEL" >&2
    docker logs qwen-vllm 2>&1 | tail -40 || true
    return 1
}

json_list_nonempty() {
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d else 1)' "$1" 2>/dev/null
}

run_named() {
    local name="$1" cmd="$2"
    docker rm -f "$name" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest MOUNTS="-v $SCRATCH:$SCRATCH:ro" \
        GPUS=all DETACH=1 NAME="$name" "$KB/scripts/in_container.sh" "$cmd"
    while docker ps --filter name="$name" --format '{{.Names}}' | grep -qx "$name"; do
        sleep 60
    done
}

IFS=',' read -r -a lv_arr <<< "$LEVELS"
for lv in "${lv_arr[@]}"; do
    lv="${lv// /}"
    [ -n "$lv" ] || continue
    case "$HELD_OUT" in
        *" $lv "*) echo "refusing held-out level $lv" >&2; exit 1 ;;
    esac
    piece="$WS/runs/rl_frontier_glc_${lv}.json"
    rollouts="$WS/runs/rl_rollouts_glc_${lv}.jsonl"
    if [ -f "$piece" ] && json_list_nonempty "$piece"; then
        echo "frontier level $lv already at $piece"
        continue
    fi
    rm -f "$piece"

    # Sample against vLLM, then stop it before verify. A second docker with
    # --gpus all cannot CUDA-init while the server is holding the cards
    # (level 86 wrote an empty frontier: 600 tasks, 0 counted).
    if [ ! -s "$rollouts" ]; then
        serve_glc || exit 1
        echo "=== sampling level $lv k=$SAMPLES ==="
        run_named "glc_front_${lv}" \
            "cd /ws/cuTileForge && python3 -u rl/select_frontier.py \
                --levels $lv --samples $SAMPLES --prompt-tier $PROMPT_TIER \
                --max-tokens 32768 --concurrency 32 --no-verify \
                --rollouts-out /ws/runs/rl_rollouts_glc_${lv}.jsonl \
                --out /ws/runs/rl_frontier_glc_${lv}.json"
        if [ ! -s "$rollouts" ]; then
            echo "sampling level $lv produced no rollouts" >&2
            docker logs "glc_front_${lv}" 2>&1 | tail -40 || true
            exit 1
        fi
    else
        echo "reusing rollouts $rollouts ($(wc -l < "$rollouts") lines)"
    fi
    docker rm -f qwen-vllm "glc_front_${lv}" >/dev/null 2>&1 || true

    echo "=== verifying level $lv ==="
    run_named "glc_front_${lv}" \
        "cd /ws/cuTileForge && python3 -u rl/select_frontier.py \
            --levels $lv --samples $SAMPLES --prompt-tier $PROMPT_TIER \
            --from-rollouts /ws/runs/rl_rollouts_glc_${lv}.jsonl \
            --verify-workers 8 --gpus 4 \
            --out /ws/runs/rl_frontier_glc_${lv}.json"
    if ! json_list_nonempty "$piece"; then
        echo "verify level $lv produced no frontier" >&2
        docker logs "glc_front_${lv}" 2>&1 | tail -40 || true
        exit 1
    fi
done

docker rm -f qwen-vllm >/dev/null 2>&1 || true
echo "screen_gl_frontier done: $LEVELS"
