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
#       Q:/path/to/model-Q
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
    MODEL="$path" MOUNTS="-v /raid/tmp:/raid/tmp:ro" \
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
