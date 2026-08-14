#!/usr/bin/env bash
# Prompt-only TILE=1024 eval: Level-1 pointwise activations plus MinGPT GELU.
#
# Matches model M's eval settings (concepts, temperature 1.0, k=4) so the
# anchors are comparable to the published 0.556x. Generation is two subset
# ranges because generate_samples.py only accepts one inclusive id interval.
set -uo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"

export CUTILE_WS="$WS"
export MODEL="${MODEL:?MODEL must point at the merged checkpoint to sample from}"
export MOUNTS="${MOUNTS:--v /raid/tmp:/raid/tmp:ro}"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
export PROMPT_TIER="${PROMPT_TIER:-cutile_concepts}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export NUM_WORKERS="${NUM_WORKERS:-32}"

RUN_NAME="${1:-tile1024_l1act}"
K="${2:-4}"
# 19-32 is every L1 activation including softmax; 88 is 8192x8192 GELU.
EXPECTED=$((15 * K))
RUN_DIR="$WS/runs/$RUN_NAME"

have() {
    python3 - "$RUN_DIR" <<'PY'
import os, sys
d = sys.argv[1]
print(sum(1 for f in os.listdir(d) if f.endswith("_kernel.py")) if os.path.isdir(d) else 0)
PY
}

server_up() { curl -s --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1; }

start_server() {
    echo "[tile1024] (re)starting vLLM"
    docker rm -f qwen-vllm >/dev/null 2>&1 || true
    "$KB/scripts/serve_qwen.sh" >/dev/null 2>&1
    for _ in $(seq 1 180); do
        server_up && { echo "[tile1024] server up"; return 0; }
        docker ps --filter name=qwen-vllm --format '{{.Names}}' | grep -q qwen-vllm || {
            echo "[tile1024] container died during startup"; return 1; }
        sleep 10
    done
    echo "[tile1024] server did not come up in time"
    return 1
}

generate_range() {
    local start="$1" end="$2"
    NAME="gen-$RUN_NAME" "$KB/scripts/run_generate.sh" "$RUN_NAME" 1 "$K" \
        "subset=($start, $end)" log_raw_response=True 2>&1 | tail -5
    docker rm -f "gen-$RUN_NAME" >/dev/null 2>&1 || true
}

echo "=== generate $RUN_NAME k=$K temp=$TEMPERATURE expected=$EXPECTED ==="
for round in $(seq 1 8); do
    got="$(have)"
    echo "=== round $round: have $got / $EXPECTED ==="
    [[ "$got" -ge "$EXPECTED" ]] && break
    server_up || start_server || { echo "cannot start server"; exit 1; }
    generate_range 19 32
    generate_range 88 88
done

got="$(have)"
if [[ "$got" -lt "$EXPECTED" ]]; then
    echo "incomplete: $got / $EXPECTED"
    exit 1
fi

echo "=== stop vLLM, verify with timing ==="
docker rm -f qwen-vllm >/dev/null 2>&1 || true

OUT="$WS/runs/${RUN_NAME}_verified.jsonl"
docker rm -f fv_tile1024 >/dev/null 2>&1 || true
CUTILE_WS="$WS" DETACH=1 NAME=fv_tile1024 "$KB/scripts/in_container.sh" \
    "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
        --kernel-dir /ws/runs/$RUN_NAME --level 1 \
        --measure-time --ref-mode compile --workers 16 --gpus 4 \
        --timeout 180 --out /ws/runs/${RUN_NAME}_verified.jsonl"

while docker ps --filter name=fv_tile1024 --format '{{.Names}}' | grep -q fv_tile1024; do
    sleep 30
done
echo "verify done: $OUT"
wc -l "$OUT" || true
