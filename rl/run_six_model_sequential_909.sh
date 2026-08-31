#!/usr/bin/env bash
# Full six-model 909-task control + 1+1+1+1 evaluation at native context.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$WS/runs/.six_model_sequential_909.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another 909 sequential run holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.six_model_sequential_909.pid"

bash "$FORGE/rl/prepare_sequential_models.sh"
# shellcheck disable=SC1090
source "$WS/runs/sequential_models.env"
bash "$FORGE/overlay/scripts/install_eval_suite.sh"

export SEQUENTIAL_USE_NATIVE_CONTEXT=1
export SEQUENTIAL_OUTPUT_CAP=131072
export SEQUENTIAL_SAFETY_MARGIN=1024

tags=(GLE GL Q base G4t Q38)
model_vars=(GLE_MODEL GL_MODEL Q_MODEL BASE_MODEL G4T_MODEL Q38_MODEL)

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    var="${model_vars[$i]}"
    model="${!var}"
    echo "=== $tag native-context two-task control smoke ==="
    bash "$FORGE/rl/run_independent_control.sh" \
        "$tag" "$model" 60 "seq128ctrl_${tag}_smoke60" 2
    echo "=== $tag native-context two-task adaptive smoke ==="
    bash "$FORGE/rl/run_sequential_search.sh" \
        "$tag" "$model" 60 "seq128_${tag}_smoke60" 2
done

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    var="${model_vars[$i]}"
    model="${!var}"
    control="seq128ctrl_${tag}_l60"
    adaptive="seq128_${tag}_l60"

    echo "=== $tag 909-task native-context independent control ==="
    bash "$FORGE/rl/run_independent_control.sh" \
        "$tag" "$model" 60 "$control"
    echo "=== $tag 909-task native-context sequential search ==="
    bash "$FORGE/rl/run_sequential_search.sh" \
        "$tag" "$model" 60 "$adaptive"
    echo "=== $tag 909-task native-context scorecard ==="
    python3 "$FORGE/verify/sequential_scorecard.py" \
        --tag "$tag" \
        --manifest "$FORGE/tasks/eval/manifest.json" \
        --level 60 \
        --control "$WS/runs/${control}_verified.jsonl" \
        --control-kernels "$WS/runs/$control" \
        --adaptive "$WS/runs/$adaptive" \
        --min-extra-solved 0 \
        --out "$WS/runs/${adaptive}_scorecard.json" \
        | tee "$WS/runs/${adaptive}_scorecard.log"
done

echo "six-model 909 native-context sequential evaluation complete"
