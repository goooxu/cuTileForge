#!/usr/bin/env bash
# Queue base / M / Q (or any TAG:PATH list) on the standalone eval suite.
#
# Each model is one run_eval_suite.sh invocation. A tag whose verified
# jsonl already exists is skipped, so a restart after an ssh drop is
# cheap. Run this detached: an attached ssh session dying takes vLLM with it.
#
# Usage:
#   CUTILE_WS=... rl/compare_eval_suite.sh \
#       base:/path/to/Qwen3-Coder-Next \
#       M:/path/to/model-M \
#       Q:/path/to/model-Q \
#       Q38:/path/to/Qwen3.8-27B \
#       Q38nt:/path/to/Qwen3.8-27B \
#       G4:/path/to/Gemma-4-31B-it \
#       G4t:/path/to/Gemma-4-31B-it \
#       GL:/path/to/Muse-Glimmer-30B
# Q38 = thinking on, max_tokens=32768. Q38nt = thinking off, max_tokens=8192.
# G4 = Gemma 4 thinking off, 8192, nightly-aarch64 (table B).
# G4t = Gemma 4 thinking on, 32768, same image (table A). No effort slider.
# Official v0.27.1-aarch64 cannot serve Gemma 4 (head_dim per-layer).
# GL = Muse Glimmer always-think, reasoning_strength=xhigh, 32768 (table A).
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
# Per-tag overrides must not leak. Restore this default at the start of each
# tag; G4 / G4t / GL then replace it.
DEFAULT_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-aarch64}"

if [[ $# -lt 1 ]]; then
    echo "usage: compare_eval_suite.sh TAG:PATH [TAG:PATH ...]" >&2
    exit 1
fi

done_tag() {
    local tag="$1"
    [[ -f "$WS/runs/${tag}_l60_verified.jsonl" ]]
}

# SKIP_INSTALL=1 when another host is already verifying from this
# checkout: install_eval_suite.sh wipes KernelBench/level60 first.
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    bash "$FORGE/overlay/scripts/install_eval_suite.sh" || {
        echo "ERROR: install_eval_suite failed" >&2
        exit 1
    }
fi

for spec in "$@"; do
    tag="${spec%%:*}"
    path="${spec#*:}"
    echo "=== $tag ($path) ==="
    if done_tag "$tag"; then
        echo "  already verified; skipping"
        continue
    fi
    if [[ ! -d "$path" ]]; then
        echo "  ERROR: model path missing: $path" >&2
        exit 1
    fi
    export MODEL="$path"
    export MOUNTS="-v /raid/tmp:/raid/tmp:ro"
    # Per-tag protocol. Always reset so a previous tag cannot leak.
    unset ENABLE_THINKING REASONING_EFFORT REASONING_STRENGTH EXTRA_ARGS || true
    export VLLM_IMAGE="$DEFAULT_IMAGE"
    export MAX_TOKENS=32768
    if [[ "$tag" == "Q38" ]]; then
        export ENABLE_THINKING=1
        export REASONING_EFFORT=xhigh
        export MAX_TOKENS=32768
    elif [[ "$tag" == "Q38nt" ]]; then
        # No-think: 32768 was only for thinking traces. Same 8192 as the
        # archived non-thinking protocol.
        export ENABLE_THINKING=0
        export MAX_TOKENS=8192
    elif [[ "$tag" == "G4" ]]; then
        # Gemma 4 official default: thinking off. Table B.
        # v0.27.1-aarch64 and v0.27.1-aarch64-cu129 both die on
        # AmbiguousGlobalPerLayerAttributeError (head_dim). Nightly works.
        export ENABLE_THINKING=0
        export MAX_TOKENS=8192
        export EXTRA_ARGS="--reasoning-parser gemma4"
        export VLLM_IMAGE=vllm/vllm-openai:nightly-aarch64
    elif [[ "$tag" == "G4t" ]]; then
        # Gemma 4 thinking on. Table A. No effort slider. Same image as G4.
        export ENABLE_THINKING=1
        export MAX_TOKENS=32768
        export EXTRA_ARGS="--reasoning-parser gemma4"
        export VLLM_IMAGE=vllm/vllm-openai:nightly-aarch64
    elif [[ "$tag" == "GL" ]]; then
        # Muse Glimmer: thinking cannot be turned off. Table A, xhigh.
        export REASONING_STRENGTH=xhigh
        export MAX_TOKENS=32768
        export EXTRA_ARGS="--reasoning-parser muse_glimmer --generation-config auto"
        export VLLM_IMAGE=vllm/vllm-openai:muse-glimmer
    fi
    bash "$FORGE/rl/run_eval_suite.sh" "$tag" || {
        echo "  ERROR: run_eval_suite failed for $tag" >&2
        exit 1
    }
done

echo "=== scorecard ==="
args=()
for spec in "$@"; do
    tag="${spec%%:*}"
    args+=(--run "$tag:$WS/runs/${tag}")
done
python3 "$FORGE/verify/eval_scorecard.py" "${args[@]}"
echo "compare complete"
