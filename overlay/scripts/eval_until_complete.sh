#!/usr/bin/env bash
# Re-run an eval until every sample has a result.
#
# The official harness does not always get through a large run in one go: a
# sample that hangs the GPU can take the worker down with it, and the run exits
# with most samples still marked not_evaluated. Those samples are silently
# excluded from pass@k, which quietly understates the run -- the k=16 control in
# phase 9 first reported 21/100 instead of 61/100 for exactly this reason.
# Eval is resumable per (problem_id, sample_id), so re-running fills the gaps.
#
# Usage: ./eval_until_complete.sh <run_name> <level> <num_samples> [max_passes]
set -uo pipefail

RUN_NAME="${1:?usage: eval_until_complete.sh <run_name> <level> <num_samples> [max_passes]}"
LEVEL="${2:?}"
NUM_SAMPLES="${3:?}"
MAX_PASSES="${4:-12}"

cd "$(dirname "$0")/.."
WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
RESULTS="$WS/runs/$RUN_NAME/eval_results.json"
EXPECTED=$(python3 -c "import sys;print(int(sys.argv[1])*int(sys.argv[2]))" \
    "$NUM_SAMPLES" "$( [ "$LEVEL" = 4 ] && echo 20 || { [ "$LEVEL" = 3 ] && echo 50 || echo 100; } )")

count_done() {
    python3 - "$RESULTS" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(sum(len(v) for v in d.values()))
except Exception:
    print(0)
PY
}

prev=-1
for pass in $(seq 1 "$MAX_PASSES"); do
    docker rm -f "ev_${RUN_NAME}" >/dev/null 2>&1
    DETACH=1 NAME="ev_${RUN_NAME}" scripts/run_eval.sh \
        "$RUN_NAME" "$LEVEL" "$NUM_SAMPLES" >/dev/null 2>&1
    while docker ps --filter "name=ev_${RUN_NAME}" --format '{{.Names}}' \
            | grep -q "ev_${RUN_NAME}"; do
        sleep 30
    done
    n=$(count_done)
    echo "pass $pass: $n / $EXPECTED evaluated"
    if [ "$n" -ge "$EXPECTED" ]; then
        echo "complete"
        exit 0
    fi
    # A pass that adds nothing means the remaining samples fail the same way
    # every time; more passes will not help.
    if [ "$n" = "$prev" ]; then
        echo "no progress this pass; $((EXPECTED - n)) samples are stuck"
        exit 1
    fi
    prev="$n"
done
echo "hit max passes with $(count_done) / $EXPECTED evaluated"
exit 1
