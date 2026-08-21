#!/usr/bin/env bash
# Harvest k=8 cuTile kernels from Muse Glimmer on a training level, keeping the
# model's own reasoning trace.
#
# The reasoning parser is deliberately left OFF here, unlike the eval protocol.
# With it, vLLM moves the reasoning channel into reasoning_content and the logged
# response keeps only the final answer -- which is why the 3636 responses from the
# GL eval run contain no trace at all. Self-distillation on an always-thinking
# model needs the trace, so the channel stays inline and extract_best_code strips
# it when it writes the kernel file (it already knows the `to=self ... <|eom|>`
# form). Sampling is otherwise the frozen protocol: concepts prompt, TILE=1024,
# reasoning_strength=xhigh, max_tokens=32768.
#
# Temperature is 1.2 rather than the eval protocol's 1.0: this is a harvest, so
# the point is coverage of distinct solutions, not a comparable score.
#
# Verification is correctness-only. Timing is a separate, exclusive-GPU phase and
# is only needed if the dataset will be filtered with --min-speedup.
#
# Usage:
#   CUTILE_WS=... rl/run_gl_harvest.sh [run_name] [level] [k]
#   SMOKE=8 CUTILE_WS=... rl/run_gl_harvest.sh   # first 8 problems, one round
set -uo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"

RUN_NAME="${1:-harvest_gl92}"
LEVEL="${2:-92}"
K="${3:-8}"
SMOKE="${SMOKE:-0}"
# Eval suite and both sealed held-out tracks. Same set as select_frontier /
# grpo.py. A harvest that writes these would leak the ruler into SFT.
case " $LEVEL " in
    *" 60 "*|*" 84 "*|*" 88 "*|*" 97 "*|*" 98 "*|*" 99 "*)
        echo "refusing held-out level $LEVEL" >&2
        exit 1
        ;;
esac

export CUTILE_WS="$WS"
export MODEL="${MODEL:-/raid/tmp/gemsg-cutile/Muse-Glimmer-30B}"
export MOUNTS="${MOUNTS:--v /raid/tmp:/raid/tmp:ro}"
# The official image's ENTRYPOINT is `vllm serve`; serve_qwen.sh branches on the
# tag for that.
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:muse-glimmer}"
export EXTRA_ARGS="${EXTRA_ARGS:---generation-config auto}"
export REASONING_STRENGTH="${REASONING_STRENGTH:-xhigh}"
# Without this the server decodes with skip_special_tokens=True and the response
# arrives as `to=self` followed by bare prose: no <|eom|> to close the channel,
# so the extractor treats every sample as truncated mid-think and writes a stub.
export KEEP_SPECIAL_TOKENS=1
export MAX_TOKENS="${MAX_TOKENS:-32768}"
export PROMPT_TIER="${PROMPT_TIER:-cutile_concepts}"
export TEMPERATURE="${TEMPERATURE:-1.2}"
export NUM_WORKERS="${NUM_WORKERS:-32}"

echo "=== generate $RUN_NAME level $LEVEL k=$K temp=$TEMPERATURE"
echo "    image=$VLLM_IMAGE strength=$REASONING_STRENGTH max_tokens=$MAX_TOKENS"
echo "    extra_args='$EXTRA_ARGS' (no reasoning parser: trace stays in the response)"

if [[ "$SMOKE" != "0" ]]; then
    echo "=== SMOKE: first $SMOKE problems, one round, not a harvest ==="
    bash "$FORGE/taskgen/generate_with_restart.sh" "$RUN_NAME" "$LEVEL" "$K" 1 \
        "subset=(1, $SMOKE)"
else
    bash "$FORGE/taskgen/generate_with_restart.sh" "$RUN_NAME" "$LEVEL" "$K" 12
fi

echo "=== stop vLLM, verify correctness ==="
docker rm -f qwen-vllm >/dev/null 2>&1 || true

KERNEL_DIR="$WS/runs/$RUN_NAME"
n_kern=0
if [[ -d "$KERNEL_DIR" ]]; then
    n_kern="$(ls "$KERNEL_DIR" | grep -c _kernel.py || true)"
fi
if [[ "$n_kern" -eq 0 ]]; then
    echo "no kernels in $KERNEL_DIR; skipping verify"
    exit 1
fi

OUT="$WS/runs/${RUN_NAME}_verified.jsonl"
FV="fv_$RUN_NAME"
docker rm -f "$FV" >/dev/null 2>&1 || true
CUTILE_WS="$WS" DETACH=1 NAME="$FV" "$KB/scripts/in_container.sh" \
    "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
        --kernel-dir /ws/runs/$RUN_NAME --level $LEVEL \
        --workers 16 --gpus 4 --timeout 180 \
        --out /ws/runs/${RUN_NAME}_verified.jsonl"

while docker ps --filter name="$FV" --format '{{.Names}}' | grep -q "$FV"; do
    sleep 60
done
docker logs "$FV" 2>&1 | tail -5 || true
echo "verify done: $OUT"
wc -l "$OUT" || true

# Timing is a second container. The correctness pool can poison CUDA; do not
# measure in the same process. Needed for a speed-gated slice (level 80).
if [[ "${MEASURE_TIME:-0}" == "1" ]]; then
    echo "=== time passed kernels vs torch.compile ==="
    TV="tv_$RUN_NAME"
    docker rm -f "$TV" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="$TV" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$RUN_NAME --level $LEVEL \
            --workers 4 --gpus 4 --timeout 180 \
            --measure-time --timing-from /ws/runs/${RUN_NAME}_verified.jsonl \
            --ref-mode compile \
            --out /ws/runs/${RUN_NAME}_verified.jsonl"
    while docker ps --filter name="$TV" --format '{{.Names}}' | grep -q "$TV"; do
        sleep 60
    done
    docker logs "$TV" 2>&1 | tail -8 || true
    echo "timing done: $OUT"
fi
