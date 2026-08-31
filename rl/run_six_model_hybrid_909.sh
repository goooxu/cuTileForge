#!/usr/bin/env bash
# Six-model 60-second Hybrid search and formal evaluation.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$WS/runs/.six_model_hybrid_909.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another six-model Hybrid run holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.six_model_hybrid_909.pid"

bash "$FORGE/rl/prepare_sequential_models.sh"
# shellcheck disable=SC1090
source "$WS/runs/sequential_models.env"
bash "$FORGE/overlay/scripts/install_eval_suite.sh"

tags=(GLE GL Q base G4t Q38)
models=(
    "$GLE_MODEL"
    "$GL_MODEL"
    "$Q_MODEL"
    "$BASE_MODEL"
    "$G4T_MODEL"
    "$Q38_MODEL"
)
windows=(32 8 32 32 8 4)

score_run() {
    local tag="$1"
    local name="$2"
    local out="$WS/runs/${name}_scorecard.json"
    python3 "$FORGE/verify/hybrid_scorecard.py" \
        --tag "$tag" --level 60 \
        --run-dir "$WS/runs/$name" \
        --manifest "$FORGE/tasks/eval/manifest.json" \
        --out "$out" \
        | tee "$WS/runs/${name}_scorecard.log"
    python3 - "$out" <<'PY'
import json
import sys
score = json.load(open(sys.argv[1]))
if score["solve"]["all"]["nonterminal"]:
    raise SystemExit("scorecard contains nonterminal tasks")
if not score["early_exit"]["audit_passed"]:
    raise SystemExit("early-exit audit failed")
PY
}

run_one() {
    local tag="$1"
    local model="$2"
    local name="$3"
    local limit="$4"
    bash "$FORGE/rl/run_hybrid_60s.sh" \
        "$tag" "$model" 60 "$name" "$limit"
    score_run "$tag" "$name"
}

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    model="${models[$i]}"
    echo "=== $tag Hybrid four-task functional smoke ==="
    run_one "$tag" "$model" "hybrid60_${tag}_smoke4" 4
done

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    model="${models[$i]}"
    window="${windows[$i]}"
    echo "=== $tag Hybrid native-window smoke: $window tasks ==="
    run_one "$tag" "$model" "hybrid60_${tag}_window${window}" "$window"
done

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    model="${models[$i]}"
    name="hybrid60_${tag}_l60"
    echo "=== $tag Hybrid 909-task 60-second search ==="
    run_one "$tag" "$model" "$name" 0
done

report_args=()
for tag in "${tags[@]}"; do
    # Regenerate every scorecard with the final schema before rendering the
    # combined report, including models completed before a resumable restart.
    score_run "$tag" "hybrid60_${tag}_l60"
    report_args+=(
        "--scorecard"
        "$tag=$WS/runs/hybrid60_${tag}_l60_scorecard.json"
    )
done
python3 "$FORGE/verify/hybrid_report.py" \
    "${report_args[@]}" \
    --out "$FORGE/results/REPORT_HYBRID_60S_909.md" \
    | tee "$WS/runs/hybrid60_report.log"

echo "six-model 909 Hybrid 60-second search evaluation complete"
