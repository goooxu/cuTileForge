#!/usr/bin/env bash
# Build/calibrate an independent tile-sensitive dev set before any module
# localisation experiment.  No weights are changed here.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE_MODEL:-/raid/tmp/gemsg-cutile/model-GLE}"
ENDPOINT="${ENDPOINT_MODEL:-/raid/tmp/gemsg-cutile/model-GLI}"
LOCK="$WS/runs/.glj_speed_dev_pipeline.lock"
HARVEST="$WS/runs/harvest_glj63_gle_verified.jsonl"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another GL-J speed-dev pipeline holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.glj_speed_dev.pipeline.pid"

timing_complete() {
    local path="$1"
    local need="$2"
    [[ -f "$path" ]] || return 1
    python3 - "$path" "$FORGE/verify" "$need" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from fast_verify import timing_complete
raise SystemExit(0 if timing_complete(sys.argv[1], need=int(sys.argv[3])) else 1)
PY
}

echo "=== rebuild frozen level-63 pool ==="
python3 "$FORGE/rl/build_glj_speed_dev_pool.py" \
    --workspace "$WS" --matmul-variants 8

if ! timing_complete "$HARVEST" 3680; then
    echo "=== GL-E k=8 harvest + timing on level 63 ==="
    MODEL="$BASE" MEASURE_TIME=1 \
        bash "$FORGE/rl/run_gl_harvest.sh" harvest_glj63_gle 63 8
else
    echo "GL-E level-63 harvest already complete"
fi
docker rm -f fv_harvest_glj63_gle tv_harvest_glj63_gle >/dev/null 2>&1 || true

echo "=== freeze tile-sensitive level 64 ==="
python3 "$FORGE/rl/freeze_glj_speed_dev.py" --workspace "$WS"

echo "=== calibrate GL-E ==="
bash "$FORGE/rl/run_glj_speed_calibration.sh" GLEC "$BASE"
echo "=== calibrate GL-I endpoint ==="
bash "$FORGE/rl/run_glj_speed_calibration.sh" GLIC "$ENDPOINT"

echo "=== speed-dev scorecard ==="
python3 "$FORGE/verify/glj_speed_calibration.py" \
    --manifest "$WS/runs/glj_speed_dev_manifest.json" \
    --base "$WS/runs/GLEC_glj_speed" \
    --candidate "$WS/runs/GLIC_glj_speed" \
    --out "$WS/runs/glj_speed_calibration.json" \
    | tee "$WS/runs/glj_speed_calibration.log"

passed="$(python3 - "$WS/runs/glj_speed_calibration.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["passed"] else "0")
PY
)"
if [[ "$passed" == "1" ]]; then
    echo "GL-J speed calibration passed; module localisation is allowed"
else
    echo "GL-J speed calibration failed; stop the speed line"
fi
echo "GL-J speed-dev pipeline complete"
