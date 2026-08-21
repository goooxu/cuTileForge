#!/usr/bin/env bash
# Run GRPO in windows, refreshing the sampler between them.
#
# grpo.py takes its rollouts from a served model that it never updates, so after
# the first iteration the data comes from a policy the trainer has already moved
# away from. The reference log-probabilities are taken from the current weights,
# not from whatever generated the sample, so nothing corrects for that gap and
# the reported clip fraction cannot see it either. The first RL run went twenty
# iterations that way and moved nothing.
#
# This bounds the gap instead of ignoring it: train for --window iterations,
# merge the adapter into a fresh servable copy, restart vLLM on it, resume. The
# sampler is then never more than one window behind the policy.
#
# Merging writes ~150 GB per refresh, so windows are kept coarse and the copies
# alternate between two slots rather than accumulating.
#
# Usage:
#   CUTILE_WS=/path/to/ws rl/refresh_loop.sh <base_model> <frontier> <out_dir> \
#       [total_iters] [window]
set -uo pipefail

BASE="${1:?usage: refresh_loop.sh <base_model> <frontier> <out_dir> [total] [window]}"
FRONTIER="${2:?}"
OUT="${3:?}"
TOTAL="${4:-60}"
WINDOW="${5:-10}"

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
SCRATCH="${RL_SCRATCH:-/raid/tmp/gemsg-cutile}"
TIER="${PROMPT_TIER:-cutile_concepts}"
GPU_UTIL="${GPU_UTIL:-0.55}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
VLLM_GPUS="${VLLM_GPUS:-all}"
TRAIN_GPUS="${TRAIN_GPUS:-all}"

TRAIN_IMAGE="cutile-train:latest"
TRAIN_MOUNTS="-v $SCRATCH:$SCRATCH"

count_done() {
    local hist="$WS/runs/$(basename "$OUT")/history.jsonl"
    [ -f "$hist" ] || { echo 0; return; }
    python3 - "$hist" <<'PY'
import json, sys
n = 0
try:
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("skipped"):
            continue
        n = max(n, rec["iteration"] + 1)
except Exception:
    n = 0
print(n)
PY
}

serve() {
    local model="$1"
    docker rm -f qwen-vllm >/dev/null 2>&1
    # After a verifier container exits, the next vLLM can see 0 CUDA devices
    # (cudaErrorNotPermitted). The frontier screen waits 20s; do the same.
    sleep 20
    CUTILE_WS="$WS" MODEL="$model" MOUNTS="-v $SCRATCH:$SCRATCH:ro" \
        GPU_UTIL="$GPU_UTIL" TENSOR_PARALLEL="$TENSOR_PARALLEL" \
        VLLM_GPUS="$VLLM_GPUS" \
        VLLM_IMAGE="${VLLM_IMAGE:-}" EXTRA_ARGS="${EXTRA_ARGS:-}" \
        "$KB/scripts/serve_qwen.sh" >/dev/null 2>&1
    for _ in $(seq 1 40); do
        if curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
                | grep -q Qwen3; then
            echo "  serving $model"
            return 0
        fi
        sleep 30
    done
    echo "  ERROR: vLLM did not come up on $model" >&2
    return 1
}

# The policy the sampler is currently serving. Starts as the unmodified base,
# which is exactly right: a fresh adapter is zero-initialised, so iteration 0 is
# genuinely on-policy.
serving="$BASE"
slot=0

# Pick up where a previous invocation stopped. Containers on this machine get
# reaped out from under long runs, so this script has to be safe to just run
# again rather than needing the iteration count passed in by hand.
done_iters=$(count_done)
if [ "$done_iters" -gt 0 ]; then
    echo "resuming after $done_iters completed iterations"
    # The adapter from the last checkpoint is the current policy, so serve that
    # rather than the base -- otherwise the restart silently reverts the sampler
    # to iteration 0 and every window after the first is maximally stale.
    for slot_try in 0 1; do
        [ -f "$SCRATCH/rl-policy-$slot_try/config.json" ] && serving="$SCRATCH/rl-policy-$slot_try"
    done
fi

while [ "$done_iters" -lt "$TOTAL" ]; do
    n=$(( TOTAL - done_iters ))
    [ "$n" -gt "$WINDOW" ] && n="$WINDOW"

    serve "$serving" || exit 1

    resume=()
    [ "$done_iters" -gt 0 ] && resume=(--resume "$OUT/ck")

    echo "=== iterations $done_iters..$(( done_iters + n - 1 )) ==="
    docker rm -f grpo >/dev/null 2>&1
    CUTILE_WS="$WS" IMAGE="$TRAIN_IMAGE" MOUNTS="$TRAIN_MOUNTS" GPUS="$TRAIN_GPUS" \
        DETACH=1 NAME=grpo "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u rl/grpo.py --model $BASE --fresh-lora \
            --prompt-tier $TIER --frontier $FRONTIER --out $OUT \
            --iterations $n --lr ${RL_LR:-3e-6} --kl-coef ${RL_KL:-0.05} \
            --temperature ${RL_TEMP:-1.0} --top-p ${RL_TOP_P:-0.95} \
            --seed ${RL_SEED:-0} \
            ${resume[*]} ${GRPO_EXTRA:-}" >/dev/null || exit 1
    while docker ps --filter name=grpo --format '{{.Names}}' | grep -q grpo; do
        sleep 60
    done
    docker logs grpo 2>&1 | grep -E "^iter|training .* parameters" | tail -"$n"

    prev=$done_iters
    done_iters=$(count_done)
    if [ "$done_iters" -le "$prev" ]; then
        echo "ERROR: GRPO window made no progress ($prev -> $done_iters)" >&2
        docker logs grpo 2>&1 | tail -40
        exit 1
    fi
    gained=$((done_iters - prev))
    if [ "$gained" -lt "$n" ] && [ "$done_iters" -lt "$TOTAL" ]; then
        echo "ERROR: GRPO window incomplete ($gained/$n iters); not merging" >&2
        docker logs grpo 2>&1 | tail -20
        exit 1
    fi
    [ "$done_iters" -ge "$TOTAL" ] && break

    # --- refresh the sampler -------------------------------------------------
    # Free the GPUs first: merging is a CPU job but vLLM would still be holding
    # memory the next training window needs.
    docker rm -f qwen-vllm >/dev/null 2>&1
    slot=$(( 1 - slot ))
    merged="$SCRATCH/rl-policy-$slot"
    echo "=== refreshing sampler into $merged ==="
    rm -rf "$merged"
    docker rm -f rlmerge >/dev/null 2>&1
    CUTILE_WS="$WS" IMAGE="$TRAIN_IMAGE" MOUNTS="$TRAIN_MOUNTS" GPUS=none \
        DETACH=1 NAME=rlmerge "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u train/merge_lora.py --base $BASE \
            --adapter $OUT/ck/adapter --out $merged" >/dev/null || exit 1
    while docker ps --filter name=rlmerge --format '{{.Names}}' | grep -q rlmerge; do
        sleep 30
    done
    if ! [ -f "$merged/config.json" ]; then
        echo "ERROR: merge produced no model in $merged" >&2
        docker logs rlmerge 2>&1 | tail -20
        exit 1
    fi
    serving="$merged"
done

echo "done: $done_iters iterations, adapter in $OUT/ck/adapter"
