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
# Any vLLM >= 0.15 image that registers Qwen3NextForCausalLM will do; override
# with VLLM_IMAGE. Must match the host architecture (this ran on aarch64).
IMAGE="${VLLM_IMAGE:?set VLLM_IMAGE to a vLLM image with Qwen3Next support}"
PORT="${PORT:-8000}"

# Prompts run ~15k tokens and completions are capped at 8k; 40k of context is
# plenty and keeps far more KV cache available than the model's native 262k.
MAX_LEN="${MAX_LEN:-40960}"

# Extra server flags. On GB200 (sm_100) the public vLLM image's inductor path
# produces an illegal memory access during profile_run, so --enforce-eager is
# needed there; NVIDIA's internal builds do not need it.
EXTRA_ARGS="${EXTRA_ARGS:---enforce-eager}"

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
