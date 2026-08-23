#!/usr/bin/env bash
# After GL-E timed harvests finish: 421-slice jsonl, SFT, merge.
# Exit 2 while harvests are still running. Exit 1 if the slice is empty/tiny.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED="${MERGED_OUT:-/raid/tmp/gemsg-cutile/model-GLF}"
mkdir -p "$WS/runs"
exec 8>"$WS/runs/.glf_pipeline.lock"
if ! flock -n 8; then
    echo "run_glf_pipeline already running"
    exit 0
fi

need=(
    "$WS/runs/harvest_gle86_verified.jsonl"
    "$WS/runs/harvest_gle87_verified.jsonl"
    "$WS/runs/harvest_gle92_verified.jsonl"
    "$WS/runs/harvest_gle93_verified.jsonl"
)
for f in "${need[@]}"; do
    if [[ ! -f "$f" ]] || [[ "$(wc -l < "$f")" -le 0 ]]; then
        echo "waiting: $f"
        exit 2
    fi
done

if ! python3 - "$FORGE/verify" "${need[@]}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from fast_verify import timing_complete
for path in sys.argv[2:]:
    if not timing_complete(path):
        raise SystemExit(1)
PY
then
    echo "waiting: GL-E harvest timing incomplete"
    exit 2
fi

if [[ -f "$MERGED/processor_config.json" ]]; then
    echo "already merged $MERGED"
    exit 0
fi

echo "=== GL-F slice (slow vs compile, kernel_ms spread) ==="
# The union harvest fails the RL band gate (median best 2.55x). Train the
# 421-problem leftover instead: best < 1.0 and kernel_ms max/min >= 1.2.

echo "=== build best-of-N SFT ==="
if [[ -f "$WS/runs/sft_glf.jsonl" ]]; then
    echo "sft jsonl exists, skip build"
else
    CUTILE_WS="$WS" IMAGE=cutile-eval:latest GPUS=none NAME=glf_mix \
        "$FORGE/kernelbench/scripts/in_container.sh" \
        "cd /ws/cuTileForge && CUTILE_WS=/ws bash rl/build_glf_sft.sh /ws/runs/sft_glf.jsonl"
fi

if [[ ! -s "$WS/runs/sft_glf.jsonl" ]]; then
    echo "best-of-N jsonl empty; not training"
    exit 1
fi
python3 - "$WS/runs/sft_glf.jsonl" <<'PY'
import json, collections, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
fam = collections.Counter(r.get("category") for r in rows)
gemm = fam.get("conv", 0) + fam.get("matmul", 0)
print("slice %d rows  GEMM %d (%.0f%%)  families %s"
      % (len(rows), gemm, 100.0 * gemm / max(len(rows), 1), dict(fam)))
if len(rows) < 80:
    raise SystemExit("slice too small")
if gemm / max(len(rows), 1) < 0.30:
    raise SystemExit("GEMM share too low")
PY

echo "=== SFT + merge ==="
bash "$FORGE/rl/run_glf_sft.sh"
echo "pipeline done: $MERGED"
