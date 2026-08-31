#!/usr/bin/env bash
# Six-model staged screen for equal-budget sequential 1+1+1+1 search.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$WS/runs/.six_model_sequential.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another six-model sequential run holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.six_model_sequential.pid"

bash "$FORGE/rl/prepare_sequential_models.sh"
# shellcheck disable=SC1090
source "$WS/runs/sequential_models.env"

python3 "$FORGE/rl/build_sequential_screen.py" --workspace "$WS"

echo "=== GLE four-task sequential smoke ==="
bash "$FORGE/rl/run_sequential_search.sh" \
    GLE "$GLE_MODEL" 65 seq_GLE_smoke65 4

tags=(GLE GL Q base G4t Q38)
model_vars=(GLE_MODEL GL_MODEL Q_MODEL BASE_MODEL G4T_MODEL Q38_MODEL)
passed_tags=()

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    var="${model_vars[$i]}"
    model="${!var}"
    control="seqctrl_${tag}_l65"
    adaptive="seq_${tag}_l65"
    echo "=== $tag 32-task independent control ==="
    bash "$FORGE/rl/run_independent_control.sh" \
        "$tag" "$model" 65 "$control"
    echo "=== $tag 32-task sequential search ==="
    bash "$FORGE/rl/run_sequential_search.sh" \
        "$tag" "$model" 65 "$adaptive"
    echo "=== $tag 32-task scorecard ==="
    python3 "$FORGE/verify/sequential_scorecard.py" \
        --tag "$tag" \
        --manifest "$WS/runs/sequential_screen_manifest.json" \
        --level 65 \
        --control "$WS/runs/${control}_verified.jsonl" \
        --control-kernels "$WS/runs/$control" \
        --adaptive "$WS/runs/$adaptive" \
        --min-extra-solved 1 \
        --out "$WS/runs/${adaptive}_scorecard.json" \
        | tee "$WS/runs/${adaptive}_scorecard.log"
    passed="$(python3 - "$WS/runs/${adaptive}_scorecard.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["gate"]["passed"] else "0")
PY
)"
    if [[ "$passed" == "1" ]]; then
        passed_tags+=("$tag")
    fi
done

printf '%s\n' "${passed_tags[@]}" \
    > "$WS/runs/sequential_screen_passed_tags.txt"
echo "screen passed tags: ${passed_tags[*]:-none}"

for tag in "${passed_tags[@]}"; do
    case "$tag" in
        GLE) model="$GLE_MODEL" ;;
        GL) model="$GL_MODEL" ;;
        Q) model="$Q_MODEL" ;;
        base) model="$BASE_MODEL" ;;
        G4t) model="$G4T_MODEL" ;;
        Q38) model="$Q38_MODEL" ;;
    esac
    control="seqctrl_${tag}_l64"
    adaptive="seq_${tag}_l64"
    echo "=== $tag full-128 independent control ==="
    bash "$FORGE/rl/run_independent_control.sh" \
        "$tag" "$model" 64 "$control"
    echo "=== $tag full-128 sequential search ==="
    bash "$FORGE/rl/run_sequential_search.sh" \
        "$tag" "$model" 64 "$adaptive"
    echo "=== $tag full-128 scorecard ==="
    python3 "$FORGE/verify/sequential_scorecard.py" \
        --tag "$tag" \
        --manifest "$WS/runs/glj_speed_dev_manifest.json" \
        --level 64 \
        --control "$WS/runs/${control}_verified.jsonl" \
        --control-kernels "$WS/runs/$control" \
        --adaptive "$WS/runs/$adaptive" \
        --min-extra-solved 2 \
        --out "$WS/runs/${adaptive}_scorecard.json" \
        | tee "$WS/runs/${adaptive}_scorecard.log"
done

echo "six-model sequential screen complete"
