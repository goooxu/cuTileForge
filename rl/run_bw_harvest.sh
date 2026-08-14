#!/usr/bin/env bash
# Harvest k=8 timed kernels on the bandwidth-bound activation set (level 83).
#
# Generation talks to vLLM; verification needs the GPUs to itself afterwards.
# Safe to re-invoke: generate_samples.py skips existing kernel files and
# fast_verify.py overwrites the jsonl.
set -uo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"

export CUTILE_WS="$WS"
export MODEL="${MODEL:?MODEL must point at the merged checkpoint to sample from}"
export MOUNTS="${MOUNTS:--v /raid/tmp:/raid/tmp:ro}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
export PROMPT_TIER="${PROMPT_TIER:-cutile_concepts}"
export TEMPERATURE="${TEMPERATURE:-1.2}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export NUM_WORKERS="${NUM_WORKERS:-32}"

RUN_NAME="${1:-harvest_bw83}"
LEVEL="${2:-83}"
K="${3:-8}"

echo "=== generate $RUN_NAME level $LEVEL k=$K temp=$TEMPERATURE ==="
bash "$FORGE/taskgen/generate_with_restart.sh" "$RUN_NAME" "$LEVEL" "$K" 16

echo "=== stop vLLM, verify with timing ==="
docker rm -f qwen-vllm >/dev/null 2>&1 || true

OUT="$WS/runs/${RUN_NAME}_verified.jsonl"
docker rm -f fv_bw >/dev/null 2>&1 || true
CUTILE_WS="$WS" DETACH=1 NAME=fv_bw "$KB/scripts/in_container.sh" \
    "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
        --kernel-dir /ws/runs/$RUN_NAME --level $LEVEL \
        --measure-time --ref-mode compile --workers 16 --gpus 4 \
        --timeout 180 --out $OUT"

while docker ps --filter name=fv_bw --format '{{.Names}}' | grep -q fv_bw; do
    sleep 60
done
echo "verify done: $OUT"
wc -l "$OUT" || true
