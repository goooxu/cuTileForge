#!/usr/bin/env bash
# Re-verify existing eval-suite kernels. Does not sample.
#
# Usage:
#   CUTILE_WS=... rl/reverify_eval_suite.sh base M Q
#   CUTILE_WS=... rl/reverify_eval_suite.sh M --level 60
set -uo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
KB="$FORGE/kernelbench"

LEVELS="60"
TAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVELS="$2"; shift 2 ;;
        *) TAGS+=("$1"); shift ;;
    esac
done
if [[ ${#TAGS[@]} -lt 1 ]]; then
    echo "usage: reverify_eval_suite.sh TAG [TAG ...] [--level 60]" >&2
    exit 1
fi

verify_one() {
    local tag="$1" level="$2"
    local run_name="${tag}_l${level}"
    local out="$WS/runs/${run_name}_verified.re.jsonl"
    local extra="--timeout 120"
    if [[ "$level" == "61" ]]; then
        extra="--measure-time --ref-mode compile --timeout 180"
    fi
    if [[ ! -d "$WS/runs/$run_name" ]]; then
        echo "ERROR: missing kernels $WS/runs/$run_name" >&2
        return 1
    fi
    echo "=== reverify $run_name -> $out ==="
    docker rm -f "fv_${run_name}_re" >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME="fv_${run_name}_re" "$KB/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/$run_name --level $level \
            --workers 16 --gpus 4 --out /ws/runs/${run_name}_verified.re.jsonl \
            $extra"
    while docker ps --filter name="fv_${run_name}_re" --format '{{.Names}}' \
            | grep -q "fv_${run_name}_re"; do
        sleep 30
    done
    if [[ ! -f "$out" ]]; then
        echo "ERROR: $out was not written" >&2
        return 1
    fi
    mv -f "$out" "$WS/runs/${run_name}_verified.jsonl"
    echo "verify done: $WS/runs/${run_name}_verified.jsonl"
}

for tag in "${TAGS[@]}"; do
    for level in $LEVELS; do
        verify_one "$tag" "$level" || exit 1
    done
done

args=()
for tag in "${TAGS[@]}"; do
    args+=(--run "$tag:$WS/runs/${tag}")
done
python3 "$FORGE/verify/eval_scorecard.py" "${args[@]}"
echo "reverify complete"
