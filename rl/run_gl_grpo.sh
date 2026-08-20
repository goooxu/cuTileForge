#!/usr/bin/env bash
# Screen a learnable frontier on GL-C, then GRPO.
#
# Glimmer's chain vs Next stopped at twin recovery (GL-C). The remaining gap
# is reliability: p@4 already beats Next-Q, p@1 does not, because too many
# problems are 1/4. GRPO is the tool for that; another distill is not.
#
# Three things that would make this a no-op if left as they were for Qwen:
#   1. train on extract_code() rather than the sampled trace
#   2. Chat dropping the reasoning channel / stripping special tokens
#   3. max_tokens=6144 against a 32768 eval protocol
# Those are fixed in grpo.py / repair_loop.py / select_frontier.py. This
# wrapper only sets the Glimmer serving env, refuses held-out levels, and
# screens per-level so a reaped job does not throw away the rest.
#
# The reasoning parser stays OFF (harvest convention). Eval turns it on.
# Do not train on 60 / 84 / 88 / 97 / 98 / 99.
#
# Usage:
#   CUTILE_WS=... rl/run_gl_grpo.sh
#   TOTAL=60 WINDOW=10 CUTILE_WS=... rl/run_gl_grpo.sh
set -uo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
SCRATCH="${RL_SCRATCH:-/raid/tmp/gemsg-cutile}"
MODEL="${MODEL:-$SCRATCH/model-GLC}"
FRONTIER_NAME="${FRONTIER_NAME:-rl_frontier_glc.json}"
OUT_NAME="${OUT_NAME:-grpo_glc}"
LEVELS="${LEVELS:-86,87,92,93}"
SAMPLES="${SAMPLES:-6}"
TOTAL="${TOTAL:-60}"
WINDOW="${WINDOW:-10}"
HELD_OUT=" 60 84 88 97 98 99 "

LOG="${GRPO_LOG:-$WS/runs/grpo_glc.log}"
LOCK="$WS/runs/.run_gl_grpo.lock"

export CUTILE_WS="$WS"
export MODEL
export MOUNTS="${MOUNTS:--v $SCRATCH:$SCRATCH}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:muse-glimmer}"
export EXTRA_ARGS="${EXTRA_ARGS:---generation-config auto}"
export REASONING_STRENGTH="${REASONING_STRENGTH:-xhigh}"
export KEEP_SPECIAL_TOKENS=1
export GPU_UTIL="${GPU_UTIL:-0.45}"
export PROMPT_TIER="${PROMPT_TIER:-cutile_concepts}"
# refresh_loop already passes --prompt-tier; do not repeat it here (argparse
# would still accept the last value, but the flag list stays one of each).
export GRPO_EXTRA="${GRPO_EXTRA:---max-tokens 32768 --max-len 20480 --gradient-checkpointing --reasoning-strength xhigh --no-speed --concurrency 32}"

mkdir -p "$WS/runs"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "run_gl_grpo.sh already holds $LOCK; exiting"
    exit 0
fi

if ! [[ "$VLLM_IMAGE" == *muse-glimmer* ]]; then
    echo "refusing VLLM_IMAGE=$VLLM_IMAGE (need muse-glimmer for GL-C)" >&2
    exit 1
fi
if ! { [ -f "$MODEL/processor_config.json" ] || [ -f "$MODEL/config.json" ]; }; then
    echo "missing $MODEL" >&2
    exit 1
fi

python3 "$FORGE/rl/test_grpo_rollout.py" || exit 1

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

IFS=',' read -r -a lv_arr <<< "$LEVELS"
pieces=()
for lv in "${lv_arr[@]}"; do
    lv="${lv// /}"
    [ -n "$lv" ] || continue
    case "$HELD_OUT" in
        *" $lv "*) echo "refusing held-out level $lv" >&2; exit 1 ;;
    esac
    piece="$WS/runs/rl_frontier_glc_${lv}.json"
    pieces+=("$piece")
    n_keep=0
    if [ -f "$piece" ]; then
        n_keep="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$piece" 2>/dev/null || echo 0)"
    fi
    if [ "${n_keep:-0}" -gt 0 ]; then
        echo "frontier level $lv already at $piece ($n_keep tasks)"
        continue
    fi
    rm -f "$piece"
    echo "=== screening level $lv k=$SAMPLES (sample, then verify without vLLM) ==="
    LEVELS="$lv" SAMPLES="$SAMPLES" CUTILE_WS="$WS" MODEL="$MODEL" \
        bash "$FORGE/rl/screen_gl_frontier.sh" || exit 1
    if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])) else 1)' "$piece"; then
        echo "select_frontier level $lv produced an empty frontier" >&2
        exit 1
    fi
done

frontier="$WS/runs/$FRONTIER_NAME"
need_merge=0
if [ ! -f "$frontier" ]; then
    need_merge=1
else
    for piece in "${pieces[@]}"; do
        if [ "$piece" -nt "$frontier" ]; then
            need_merge=1
            break
        fi
    done
fi
if [ "$need_merge" = 1 ]; then
    ins=()
    for piece in "${pieces[@]}"; do
        ins+=(--in "$piece")
    done
    python3 "$FORGE/rl/merge_frontier.py" "${ins[@]}" --out "$frontier"
fi
echo "frontier: $frontier ($(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$frontier") tasks)"

# refresh_loop serves itself; a leftover screen server would steal GPUs.
docker rm -f qwen-vllm >/dev/null 2>&1 || true

echo "=== GRPO $TOTAL iterations, window $WINDOW ==="
exec bash "$FORGE/rl/supervise_grpo.sh" \
    "$MODEL" "/ws/runs/$FRONTIER_NAME" "/ws/runs/$OUT_NAME" \
    "$TOTAL" "$WINDOW"
