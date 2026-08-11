#!/usr/bin/env bash
# Sample several models over the sealed held-out levels, then verify everything.
#
# A single number on the held-out set says nothing: the synthetic tasks are much
# easier than KernelBench (60% per-sample pass rate against roughly 21% on the
# benchmark), so what makes this a generalisation check is whether the ordering
# and the relative spacing between models carry over from the dev set. That means
# every model has to be measured in the same pass, which is what this script is
# for.
#
# Serving is the part that keeps failing: an ssh drop during vLLM startup takes
# the server with it, and there are three startups here. So run this detached and
# poll the log.
#
# Usage:
#   CUTILE_WS=... scripts/heldout_sweep.sh "base:/raid/.../Qwen3-Coder-Next" \
#       "F:/raid/.../model-F" "K:/raid/.../model-K"
set -uo pipefail

cd "$(dirname "$0")/.."
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
LEVELS="${HO_LEVELS:-97 98}"
K="${HO_K:-4}"
TIER="${PROMPT_TIER:-cutile_concepts}"
SCRATCH_MOUNT="-v /raid/tmp:/raid/tmp:ro"

serve() {
    local path="$1"
    docker rm -f qwen-vllm >/dev/null 2>&1
    sleep 5
    CUTILE_WS="$WS" MODEL="$path" MOUNTS="$SCRATCH_MOUNT" GPU_UTIL=0.85 \
        scripts/serve_qwen.sh >/dev/null 2>&1
    for _ in $(seq 1 40); do
        if curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
                | grep -q Qwen3; then
            return 0
        fi
        sleep 30
    done
    return 1
}

n_samples() {
    # Count via the generation log rather than listing the directory: these live
    # on NFS and a directory listing of thousands of files times out.
    docker logs "$1" 2>&1 | tr '\r' '\n' | grep -cE "^Generated" || true
}

for spec in "$@"; do
    tag="${spec%%:*}"
    path="${spec#*:}"
    echo "=== $tag ($path) ==="

    # Skip a model whose runs are already complete, so a restart is cheap.
    need=0
    for lvl in $LEVELS; do
        [ -d "$WS/runs/ho_${tag}_l${lvl}" ] || need=1
    done
    if [ "$need" = "0" ]; then
        echo "  already sampled; skipping"
        continue
    fi

    if ! serve "$path"; then
        echo "  ERROR: vLLM did not come up for $tag" >&2
        exit 1
    fi
    echo "  serving"

    for lvl in $LEVELS; do
        name="ho_${tag}_l${lvl}"
        docker rm -f "gen_$name" >/dev/null 2>&1
        CUTILE_WS="$WS" PROMPT_TIER="$TIER" DETACH=1 NAME="gen_$name" \
            scripts/run_generate.sh "$name" "$lvl" "$K" >/dev/null 2>&1
        while docker ps --filter "name=gen_$name" --format '{{.Names}}' \
                | grep -q "gen_$name"; do
            sleep 30
        done
        docker logs "gen_$name" 2>&1 | tr '\r' '\n' \
            | grep -E "^Generated" | tail -1 | sed "s/^/  level$lvl: /"
    done
done

# Verification wants the GPUs to itself.
docker rm -f qwen-vllm >/dev/null 2>&1
sleep 8

for spec in "$@"; do
    tag="${spec%%:*}"
    for lvl in $LEVELS; do
        name="ho_${tag}_l${lvl}"
        out="$WS/runs/$name/verified.jsonl"
        if [ -f "$out" ]; then
            echo "  $name already verified"
            continue
        fi
        docker rm -f "hv_$name" >/dev/null 2>&1
        CUTILE_WS="$WS" GPUS=all DETACH=1 NAME="hv_$name" \
            scripts/in_container.sh \
            "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
                --kernel-dir /ws/runs/$name --level $lvl \
                --out /ws/runs/$name/verified.jsonl --workers 16 --gpus 4" \
            >/dev/null 2>&1
        while docker ps --filter "name=hv_$name" --format '{{.Names}}' \
                | grep -q "hv_$name"; do
            sleep 30
        done
        docker logs "hv_$name" 2>&1 | grep -E "^done:" | sed "s/^/  $name /"
    done
done

echo "sweep complete"
