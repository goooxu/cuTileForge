#!/usr/bin/env bash
# SFT + merge GL-F from GL-E. Speed-only best-of-N, no distill mix.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
BASE="${MODEL:-/raid/tmp/gemsg-cutile/model-GLE}"
DATA="${SFT_DATA:-$WS/runs/sft_glf.jsonl}"
ADAPTER="${ADAPTER_OUT:-$WS/models/lora-GLF}"
MERGED="${MERGED_OUT:-/raid/tmp/gemsg-cutile/model-GLF}"

as_container() {
    local p="$1"
    if [[ "$p" == "$WS"/* ]]; then
        echo "/ws/${p#"$WS"/}"
    else
        echo "$p"
    fi
}
DATA_C="$(as_container "$DATA")"
ADAPTER_C="$(as_container "$ADAPTER")"

if [[ ! -f "$DATA" ]]; then
    echo "missing $DATA; run rl/build_glf_sft.sh first" >&2
    exit 1
fi

export CUTILE_WS="$WS"
export IMAGE="${IMAGE:-cutile-train:latest}"
export MOUNTS="${MOUNTS:--v /raid/tmp:/raid/tmp}"

echo "=== SFT GL-F from $BASE ==="
CUTILE_WS="$WS" NAME=glf_sft "$KB/scripts/in_container.sh" \
    "cd /ws/cuTileForge && python3 -u train/train_lora.py \
        --model $BASE --data $DATA_C --out $ADAPTER_C \
        --epochs 2 --lr 5e-5 --lora-r 128 --max-len 20480 \
        --targets attention_only --gradient-checkpointing \
        --reasoning-strength xhigh --on-overflow drop"

echo "=== merge $ADAPTER onto $BASE -> $MERGED ==="
CUTILE_WS="$WS" NAME=glf_merge MOUNTS="-v /raid/tmp:/raid/tmp" GPUS=none \
    "$KB/scripts/in_container.sh" \
    "cd /ws/cuTileForge && python3 -u train/merge_lora.py \
        --base $BASE --adapter $ADAPTER_C --out $MERGED"
echo "merged $MERGED"
