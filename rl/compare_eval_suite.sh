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
#       G4:/path/to/Gemma-4-31B-it
# Q38 = thinking on, max_tokens=32768. Q38nt = thinking off, max_tokens=8192.
# G4 = Gemma 4 official default: thinking off, max_tokens=8192 (table B).
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"

if [[ $# -lt 1 ]]; then
    echo "usage: compare_eval_suite.sh TAG:PATH [TAG:PATH ...]" >&2
    exit 1
fi

done_tag() {
    local tag="$1"
    [[ -f "$WS/runs/${tag}_l60_verified.jsonl" ]]
}

bash "$FORGE/overlay/scripts/install_eval_suite.sh" || {
    echo "ERROR: install_eval_suite failed" >&2
    exit 1
}

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
    # Per-tag protocol. Always set MAX_TOKENS so a previous tag cannot leak.
    if [[ "$tag" == "Q38" ]]; then
        export ENABLE_THINKING=1
        export REASONING_EFFORT=xhigh
        export MAX_TOKENS=32768
        unset EXTRA_ARGS || true
    elif [[ "$tag" == "Q38nt" ]]; then
        # No-think: 32768 was only for thinking traces. Same 8192 as the
        # archived non-thinking protocol.
        export ENABLE_THINKING=0
        unset REASONING_EFFORT || true
        export MAX_TOKENS=8192
        unset EXTRA_ARGS || true
    elif [[ "$tag" == "G4" ]]; then
        # Official Gemma 4 default: thinking off. Table B only.
        export ENABLE_THINKING=0
        unset REASONING_EFFORT || true
        export MAX_TOKENS=8192
        export EXTRA_ARGS="--reasoning-parser gemma4"
    else
        unset ENABLE_THINKING REASONING_EFFORT EXTRA_ARGS || true
        export MAX_TOKENS=32768
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
