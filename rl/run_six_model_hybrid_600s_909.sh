#!/usr/bin/env bash
# Six-model ten-minute Hybrid search followed by formal evaluation.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$WS/runs/.six_model_hybrid_600s_909.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another six-model Hybrid-600s run holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.six_model_hybrid_600s_909.pid"

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
if score["search_telemetry"]["terminal"]["nonterminal"]:
    raise SystemExit("scorecard contains nonterminal search tasks")
if not score["early_exit"]["audit_passed"]:
    raise SystemExit("early-exit audit failed")
PY
}

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    model="${models[$i]}"
    name="hybrid600_${tag}_smoke4"
    echo "=== $tag Hybrid-600s four-task smoke ==="
    bash "$FORGE/rl/run_hybrid_600s.sh" \
        "$tag" "$model" 60 "$name" 4
    score_run "$tag" "$name"
done

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    model="${models[$i]}"
    name="hybrid600_${tag}_l60"
    echo "=== $tag Hybrid-600s 909-task search ==="
    bash "$FORGE/rl/run_hybrid_600s.sh" \
        "$tag" "$model" 60 "$name"
    score_run "$tag" "$name"
done

python3 "$FORGE/verify/hybrid_600s_report.py" \
    --runs-root "$WS/runs" \
    --out "$FORGE/results/REPORT_HYBRID_600S_909.md" \
    | tee "$WS/runs/hybrid600_report.log"

echo "six-model 909 Hybrid-600s search evaluation complete"
