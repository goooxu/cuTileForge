#!/usr/bin/env bash
# Select a scalar multiple of GL-I's LoRA delta on a non-eval probe, then run
# table A once only if a candidate passes the frozen probe gates.
set -euo pipefail
ulimit -c 0

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KB="$FORGE/kernelbench"
BASE="${BASE_MODEL:-/raid/tmp/gemsg-cutile/model-GLE}"
ENDPOINT="${ENDPOINT_MODEL:-/raid/tmp/gemsg-cutile/model-GLI}"
SOURCE_ADAPTER="${SOURCE_ADAPTER:-$WS/models/lora-GLI}"
RAID_ROOT="${RAID_ROOT:-/raid/tmp/gemsg-cutile}"
LOCK="$WS/runs/.glj_alpha_pipeline.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another GL-J alpha pipeline holds $LOCK" >&2
    exit 1
fi
echo "$$" > "$WS/runs/.glj_alpha.pipeline.pid"

tags=(GLJa25 GLJa40 GLJa55 GLJa70)
scales=(0.25 0.40 0.55 0.70)

probe_done() {
    local tag="$1"
    local out="$WS/runs/${tag}_glj_probe_verified.jsonl"
    [[ -f "$out" ]] || return 1
    python3 - "$out" "$FORGE/verify" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from fast_verify import timing_complete
raise SystemExit(0 if timing_complete(sys.argv[1], need=256) else 1)
PY
}

merge_candidate() {
    local tag="$1"
    local scale="$2"
    local adapter="$WS/models/lora-$tag"
    local merged="$RAID_ROOT/model-$tag"
    python3 "$FORGE/train/scale_lora.py" \
        --adapter "$SOURCE_ADAPTER" --out "$adapter" --scale "$scale"
    if [[ ! -d "$merged" ]]; then
        echo "=== merge $tag scale=$scale ==="
        CUTILE_WS="$WS" IMAGE=cutile-train:latest NAME="merge_${tag}" \
            MOUNTS="-v /raid/tmp:/raid/tmp" GPUS=none \
            "$KB/scripts/in_container.sh" \
            "cd /ws/cuTileForge && python3 -u train/merge_lora.py \
                --base $BASE --adapter /ws/models/lora-$tag \
                --out $merged"
    fi
}

echo "=== freeze GL-J probe before candidates ==="
python3 "$FORGE/rl/build_glj_probe.py" --workspace "$WS"

echo "=== probe baseline GL-E ==="
bash "$FORGE/rl/run_glj_probe.sh" GLEP "$BASE"

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    scale="${scales[$i]}"
    if ! probe_done "$tag"; then
        merge_candidate "$tag" "$scale"
        bash "$FORGE/rl/run_glj_probe.sh" "$tag" "$RAID_ROOT/model-$tag"
    else
        echo "$tag probe already complete"
    fi
done

echo "=== probe GL-I endpoint control ==="
bash "$FORGE/rl/run_glj_probe.sh" GLIP "$ENDPOINT"

echo "=== frozen probe scorecard ==="
score_args=(
    --manifest "$WS/runs/glj_probe_manifest.json"
    --base "GLEP:$WS/runs/GLEP_glj_probe"
)
for i in "${!tags[@]}"; do
    score_args+=(--candidate "${tags[$i]}:${scales[$i]}:$WS/runs/${tags[$i]}_glj_probe")
done
score_args+=(--control "GLIP:$WS/runs/GLIP_glj_probe")
score_args+=(--out "$WS/runs/glj_probe_scorecard.json")
python3 "$FORGE/verify/glj_probe_scorecard.py" "${score_args[@]}" \
    | tee "$WS/runs/glj_probe_scorecard.log"

selected="$(python3 - "$WS/runs/glj_probe_scorecard.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1])).get("selected")
print(row["tag"] if row else "")
PY
)"

if [[ -z "$selected" ]]; then
    echo "no alpha passed the frozen probe; no table A"
    for tag in "${tags[@]}"; do
        rm -rf "$RAID_ROOT/model-$tag" "$WS/models/lora-$tag"
    done
    echo "GL-J alpha pipeline stopped at probe"
    exit 0
fi

echo "=== selected $selected; keep one merged checkpoint ==="
rm -rf "$RAID_ROOT/model-GLJ" "$WS/models/lora-GLJ"
mv "$RAID_ROOT/model-$selected" "$RAID_ROOT/model-GLJ"
mv "$WS/models/lora-$selected" "$WS/models/lora-GLJ"
for tag in "${tags[@]}"; do
    [[ "$tag" == "$selected" ]] && continue
    rm -rf "$RAID_ROOT/model-$tag" "$WS/models/lora-$tag"
done

echo "=== GL-J table A ==="
bash "$FORGE/rl/compare_eval_suite.sh" \
    "GLJ:$RAID_ROOT/model-GLJ" | tee "$WS/runs/eval_GLJ.log"

echo "=== GL-J pairwise against published and fastest controls ==="
python3 "$FORGE/verify/eval_scorecard.py" \
    --run "GLE:$WS/runs/GLE" \
    --run "GLH:$WS/runs/GLH" \
    --run "GLI:$WS/runs/GLI" \
    --run "GLJ:$WS/runs/GLJ" \
    | tee "$WS/runs/glj_table_a_scorecard.log"
echo "GL-J alpha pipeline complete"
