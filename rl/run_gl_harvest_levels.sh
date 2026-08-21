#!/usr/bin/env bash
# Run run_gl_harvest.sh over several training levels, skipping a level whose
# verified jsonl is already there. Held-out levels are refused here and again
# inside run_gl_harvest.sh.
#
# Usage:
#   CUTILE_WS=... MODEL=/raid/tmp/gemsg-cutile/model-GLC \
#     rl/run_gl_harvest_levels.sh harvest_glc 86,87,92,93 8
set -uo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${1:?usage: run_gl_harvest_levels.sh PREFIX LEVELS [k]}"
LEVELS="${2:?}"
K="${3:-8}"
HELD=" 60 84 88 97 98 99 "

echo "=== harvest prefix=$PREFIX levels=$LEVELS k=$K model=${MODEL:-} ==="
IFS=',' read -r -a lv_arr <<< "$LEVELS"
for lv in "${lv_arr[@]}"; do
    lv="$(echo "$lv" | tr -d '[:space:]')"
    [[ -n "$lv" ]] || continue
    if [[ "$HELD" == *" $lv "* ]]; then
        echo "refusing held-out level $lv" >&2
        exit 1
    fi
    run="${PREFIX}${lv}"
    out="$WS/runs/${run}_verified.jsonl"
    if [[ -f "$out" ]] && [[ "$(wc -l < "$out")" -gt 0 ]]; then
        if [[ "${MEASURE_TIME:-0}" != "1" ]]; then
            echo "=== $run already verified ($(wc -l < "$out") lines); skip ==="
            continue
        fi
        if python3 - "$out" "$FORGE/verify" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from fast_verify import timing_complete
sys.exit(0 if timing_complete(sys.argv[1]) else 1)
PY
        then
            echo "=== $run already timed; skip ==="
            continue
        fi
        echo "=== $run verified but untimed; harvest will add timing ==="
    fi
    echo "=== $run level $lv ==="
    bash "$FORGE/rl/run_gl_harvest.sh" "$run" "$lv" "$K" || {
        echo "ERROR: harvest failed for $run" >&2
        exit 1
    }
done
echo "harvest levels done: $PREFIX $LEVELS"
