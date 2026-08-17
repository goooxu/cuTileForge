#!/usr/bin/env bash
# Wait for fv_G4t_l60, retry timing once if jsonl has no speedups, then scorecard.
set -uo pipefail
WS="${CUTILE_WS:?}"
FORGE="$WS/cuTileForge"
LOG="$WS/runs/g4t_timing.log"
exec >>"$LOG" 2>&1
echo "waiter $(date -Is)"

wait_fv() {
    while docker ps --filter name=fv_G4t_l60 --format '{{.Names}}' | grep -q fv_G4t_l60; do
        sleep 30
    done
}

speedups() {
    python3 - "$WS/runs/G4t_l60_verified.jsonl" <<'PY'
import json, sys
print(sum(1 for line in open(sys.argv[1]) if json.loads(line).get("speedup")))
PY
}

wait_fv
echo "exited $(date -Is) speedups=$(speedups)"
docker logs fv_G4t_l60 2>&1 | tail -40 || true

if [[ "$(speedups)" == "0" ]]; then
    echo "retry timing"
    docker rm -f fv_G4t_l60 >/dev/null 2>&1 || true
    CUTILE_WS="$WS" DETACH=1 NAME=fv_G4t_l60 \
        "$FORGE/kernelbench/scripts/in_container.sh" \
        "cd /ws/cuTileForge && python3 -u verify/fast_verify.py \
            --kernel-dir /ws/runs/G4t_l60 --level 60 \
            --workers 4 --gpus 4 \
            --out /ws/runs/G4t_l60_verified.jsonl \
            --measure-time --timing-from /ws/runs/G4t_l60_verified.jsonl \
            --ref-mode compile --timeout 180"
    wait_fv
    echo "retry exited $(date -Is) speedups=$(speedups)"
fi

python3 "$FORGE/verify/eval_scorecard.py" \
    --run "base:$WS/runs/base" --run "M:$WS/runs/M" --run "Q:$WS/runs/Q" \
    --run "Q38:$WS/runs/Q38" --run "GL:$WS/runs/GL" --run "G4t:$WS/runs/G4t" \
    --k 4 || echo scorecard_failed
echo "waiter done $(date -Is)"
