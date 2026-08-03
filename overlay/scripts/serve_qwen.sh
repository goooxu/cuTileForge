#!/usr/bin/env bash
# Serve Qwen3-Coder-Next with vLLM for the generation phase.
#
# Uses the arm64 vLLM 0.26 image already on the dev machine, which registers
# Qwen3NextForCausalLM (the hybrid Gated DeltaNet + MoE architecture). BF16
# weights are ~159 GB, so TP=4 leaves ample room for KV cache on 4x189 GB.
#
# Generation and kernel evaluation run in separate phases, so this takes all
# four GPUs; stop it before running eval_from_generations.py.
set -euo pipefail

WS="${CUTILE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Any vLLM image that registers Qwen3NextForCausalLM will do, but it must match
# the host architecture. The v0.26.0 release image crashes on GB200 during
# startup (illegal memory access in FlashInfer's autotune warmup) and needs both
# a patched image and --enforce-eager; the nightly fixes this, runs unpatched
# with CUDA graphs, and is ~60x faster in decode as a result.
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
PORT="${PORT:-8000}"

# Prompts run ~15k tokens and completions are capped at 8k; 40k of context is
# plenty and keeps far more KV cache available than the model's native 262k.
MAX_LEN="${MAX_LEN:-40960}"

# Extra server flags. Empty by default: the nightly needs no workarounds on
# GB200. Set EXTRA_ARGS=--enforce-eager if pinning to an older image that does.
EXTRA_ARGS="${EXTRA_ARGS:-}"

docker rm -f qwen-vllm 2>/dev/null || true

# Runs as root deliberately. FlashInfer resolves its cubin cache to a path inside
# system dist-packages (the flashinfer_cubin package takes priority over
# FLASHINFER_CUBIN_DIR), so a non-root user cannot populate it and engine startup
# fails during profile_run. The workspace is only read here, and root can read it
# through the NFS root-squash, so nothing needs to be written back as root.
docker run -d --name qwen-vllm \
    --gpus all --ipc=host --network host \
    -e HOME=/tmp \
    -e HF_HOME=/tmp/hf \
    -e VLLM_CACHE_ROOT=/tmp/vllm-cache \
    -v "$WS":"$WS":ro \
    --entrypoint python3 \
    "$IMAGE" -m vllm.entrypoints.openai.api_server \
        --model "$WS/models/Qwen3-Coder-Next" \
        --served-model-name Qwen3-Coder-Next \
        --tensor-parallel-size 4 \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --port "$PORT" \
        --host 0.0.0.0 \
        $EXTRA_ARGS

echo "started container qwen-vllm on port $PORT"
echo "follow startup with: docker logs -f qwen-vllm"
