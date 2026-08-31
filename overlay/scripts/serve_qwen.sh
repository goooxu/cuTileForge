#!/usr/bin/env bash
# Serve Qwen3-Coder-Next with vLLM for the generation phase.
#
# Uses the arm64 vLLM 0.26 image already on the dev machine, which registers
# Qwen3NextForCausalLM (the hybrid Gated DeltaNet + MoE architecture). BF16
# weights are ~159 GB, so TP=4 leaves ample room for KV cache on 4x189 GB.
#
# Batch generation and evaluation run in separate phases, so by default this
# takes all four GPUs; stop it before running eval_from_generations.py. The
# repair loop is the exception -- it verifies while the server is up -- so set
# GPU_UTIL well below the default there to leave the verifier room.
set -euo pipefail

WS="${CUTILE_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Any vLLM image that registers Qwen3NextForCausalLM will do, but it must match
# the host architecture. The v0.26.0 release image crashes on GB200 during
# startup (illegal memory access in FlashInfer's autotune warmup) and needs both
# a patched image and --enforce-eager; the nightly fixes this, runs unpatched
# with CUDA graphs, and is ~60x faster in decode as a result.
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
PORT="${PORT:-8000}"

# Prompts run ~2.4k (cutile_concepts) and completions are capped at 32k; 40k
# of context is enough and keeps far more KV cache than the native 262k.
MAX_LEN="${MAX_LEN:-40960}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
# "all" or a docker device list such as device=0,1 so the trainer can own the
# remaining cards. device=N,M must be passed as the quoted form
# --gpus "device=N,M"; the unquoted form sets Count and DeviceIDs together
# and nvidia-container-toolkit leaves the container in created/exit 128.
VLLM_GPUS="${VLLM_GPUS:-all}"
docker_gpus_flag() {
    case "$1" in
        device=*) printf '"%s"' "$1" ;;
        *) printf '%s' "$1" ;;
    esac
}

# Fraction of each GPU vLLM may claim. 0.90 suits a phase that owns the machine.
# Anything sharing the GPUs concurrently -- the repair loop's verifier workers --
# needs this lowered or they will OOM against the server's reservation; 0.55
# leaves ~82 GB per GPU and is what the repair runs used.
GPU_UTIL="${GPU_UTIL:-0.90}"

# Which weights to serve, and anything extra to mount to reach them. Fine-tuned
# models are merged onto local NVMe rather than the workspace, so serving one
# means pointing both of these at /raid.
MODEL="${MODEL:-$WS/models/Qwen3-Coder-Next}"
read -r -a extra_mounts <<< "${MOUNTS:-}"

# Extra server flags. Empty by default: the nightly needs no workarounds on
# GB200. Set EXTRA_ARGS=--enforce-eager if pinning to an older image that does.
EXTRA_ARGS="${EXTRA_ARGS:-}"

docker rm -f qwen-vllm 2>/dev/null || true

# Runs as root deliberately. FlashInfer resolves its cubin cache to a path inside
# system dist-packages (the flashinfer_cubin package takes priority over
# FLASHINFER_CUBIN_DIR), so a non-root user cannot populate it and engine startup
# fails during profile_run. The workspace is only read here, and root can read it
# through the NFS root-squash, so nothing needs to be written back as root.
#
# The official muse-glimmer image ENTRYPOINT is `vllm serve`; the python -m
# module path is not what that image is built around.
common_args=(
    --gpus "$(docker_gpus_flag "$VLLM_GPUS")" --ipc=host --network host
    -e HOME=/tmp
    -e HF_HOME=/tmp/hf
    -e VLLM_CACHE_ROOT=/tmp/vllm-cache
    -e VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="${VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS:-}"
    -v "$WS":"$WS":ro
)
[[ ${#extra_mounts[@]} -gt 0 ]] && common_args+=("${extra_mounts[@]}")
serve_flags=(
    --model "$MODEL"
    --served-model-name Qwen3-Coder-Next
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --enable-prefix-caching
    --port "$PORT"
    --host 0.0.0.0
)
if [[ "$IMAGE" == *muse-glimmer* ]]; then
    docker run -d --name qwen-vllm "${common_args[@]}" \
        --entrypoint vllm \
        "$IMAGE" serve "${serve_flags[@]}" $EXTRA_ARGS
else
    docker run -d --name qwen-vllm "${common_args[@]}" \
        --entrypoint python3 \
        "$IMAGE" -m vllm.entrypoints.openai.api_server \
        "${serve_flags[@]}" $EXTRA_ARGS
fi

echo "started container qwen-vllm on port $PORT"
echo "follow startup with: docker logs -f qwen-vllm"
