#!/usr/bin/env bash
# After both GL-E harvests finish: gate, mix, SFT, merge.
# Does nothing (exit 2) while harvests are still running.
# Exits 1 if the speed gate fails -- do not train a distill-only stand-in.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED="${MERGED_OUT:-/raid/tmp/gemsg-cutile/model-GLE}"
mkdir -p "$WS/runs"
exec 8>"$WS/runs/.gle_pipeline.lock"
if ! flock -n 8; then
    echo "run_gle_pipeline already running"
    exit 0
fi

need=(
    "$WS/runs/harvest_glc86_verified.jsonl"
    "$WS/runs/harvest_glc87_verified.jsonl"
    "$WS/runs/harvest_glc92_verified.jsonl"
    "$WS/runs/harvest_glc93_verified.jsonl"
    "$WS/runs/harvest_glc80_verified.jsonl"
)
for f in "${need[@]}"; do
    if [[ ! -f "$f" ]] || [[ "$(wc -l < "$f")" -le 0 ]]; then
        echo "waiting: $f"
        exit 2
    fi
done

if ! python3 - "$WS/runs/harvest_glc80_verified.jsonl" "$FORGE/verify" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from fast_verify import timing_complete
sys.exit(0 if timing_complete(sys.argv[1]) else 1)
PY
then
    echo "waiting: level 80 timing incomplete"
    exit 2
fi

if [[ -f "$MERGED/processor_config.json" ]]; then
    echo "already merged $MERGED"
    exit 0
fi

echo "=== speed gate ==="
python3 "$FORGE/rl/speed_gate.py" \
    --verified "$WS/runs/harvest_glc80_verified.jsonl" --level 80 \
    | tee "$WS/runs/gle_speed_gate.txt"
# speed_gate exits 1 on fail; pipe hides that, so re-run for the status.
if ! python3 "$FORGE/rl/speed_gate.py" \
        --verified "$WS/runs/harvest_glc80_verified.jsonl" --level 80 \
        >/dev/null; then
    echo "GATE failed; not mixing a distill-only set"
    exit 1
fi

echo "=== mix SFT ==="
if [[ -f "$WS/runs/sft_gle.jsonl" ]]; then
    echo "sft jsonl exists, skip mix"
else
    # Host python on the GPU box has no kernelbench extras (tqdm). Build
    # the jsonl in the eval image; CUTILE_WS inside the container is /ws.
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=none NAME=gle_mix \
        "$FORGE/kernelbench/scripts/in_container.sh" \
        "cd /ws/cuTileForge && CUTILE_WS=/ws bash rl/mix_gl_e_sft.sh /ws/runs/sft_gle.jsonl"
fi

echo "=== SFT + merge ==="
bash "$FORGE/rl/run_gle_sft.sh"
echo "pipeline done: $MERGED"
