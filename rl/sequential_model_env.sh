#!/usr/bin/env bash
# Per-model serving protocol for equal-budget sequential-search experiments.

configure_sequential_model() {
    local tag="$1"
    local model="$2"

    export MODEL="$model"
    export MOUNTS="-v /raid/tmp:/raid/tmp:ro"
    export VLLM_IMAGE="${DEFAULT_VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
    export MAX_TOKENS="${SEQUENTIAL_OUTPUT_CAP:-32768}"
    export MAX_LEN=65536
    export NATIVE_CONTEXT=262144
    export GPU_UTIL="${GPU_UTIL:-0.90}"
    export TENSOR_PARALLEL=4
    export KEEP_SPECIAL_TOKENS=1
    unset ENABLE_THINKING REASONING_EFFORT REASONING_STRENGTH EXTRA_ARGS \
        VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS || true

    case "$tag" in
        GLE|GL)
            export NATIVE_CONTEXT=131072
            export VLLM_IMAGE=vllm/vllm-openai:muse-glimmer
            export REASONING_STRENGTH=xhigh
            export EXTRA_ARGS="--reasoning-parser muse_glimmer --generation-config auto"
            ;;
        Q38)
            export ENABLE_THINKING=1
            export REASONING_EFFORT=xhigh
            # A verified Q38 candidate can itself exceed 32K tokens. The next
            # round keeps that complete code plus a 32K output budget.
            export MAX_LEN=98304
            ;;
        G4t)
            export ENABLE_THINKING=1
            export EXTRA_ARGS="--reasoning-parser gemma4"
            ;;
        Q|base)
            # The current nightly's SM100 trtllm_bf16_moe autotuner can hit an
            # illegal address even when autotuning is skipped. Force the
            # unquantized Triton MoE backend for both the control and adaptive
            # arms; decoding semantics stay matched within each model.
            export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=trtllm_bf16_moe
            export EXTRA_ARGS="--moe-backend triton"
            ;;
        *)
            echo "unsupported sequential-search tag: $tag" >&2
            return 1
            ;;
    esac

    if [[ "${SEQUENTIAL_USE_NATIVE_CONTEXT:-0}" == "1" ]]; then
        export MAX_LEN="$NATIVE_CONTEXT"
    fi
}
