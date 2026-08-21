#!/usr/bin/env bash
# Build the mixed GL-E SFT jsonl from GL-C distill harvests + level 80 speed.
#
# Distill: response traces, cutile_concepts, max 3 / problem, GL-C quotas.
# Speed: timed catchable band, 20-35% of the finished set.
set -euo pipefail

WS="${CUTILE_WS:?CUTILE_WS must point at the workspace root}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$WS/runs/sft_gle.jsonl}"
DISTILL_OUT="$WS/runs/sft_gle_distill.jsonl"
SPEED_OUT="$WS/runs/sft_gle_speed.jsonl"

cd "$FORGE"
python3 train/build_sft_dataset.py \
    --run 86:$WS/runs/harvest_glc86:$WS/runs/harvest_glc86_verified.jsonl \
    --run 87:$WS/runs/harvest_glc87:$WS/runs/harvest_glc87_verified.jsonl \
    --run 92:$WS/runs/harvest_glc92:$WS/runs/harvest_glc92_verified.jsonl \
    --run 93:$WS/runs/harvest_glc93:$WS/runs/harvest_glc93_verified.jsonl \
    --completion-from response --prompt-tier cutile_concepts \
    --max-per-problem 3 \
    --category-quota norm=400,loss=200 \
    --out "$DISTILL_OUT"

python3 train/build_sft_dataset.py \
    --run 80:$WS/runs/harvest_glc80:$WS/runs/harvest_glc80_verified.jsonl \
    --completion-from response --prompt-tier cutile_concepts \
    --max-per-problem 3 \
    --require-timing --min-speedup 0.40 --max-speedup 1.05 \
    --out "$SPEED_OUT"

python3 - "$DISTILL_OUT" "$SPEED_OUT" "$OUT" <<'PY'
import json, random, sys
distill = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
speed = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
if not speed:
    raise SystemExit("speed slice empty; do not mix a distill-only set")
# Target 20-35% speed. If distill is huge, downsample it by problem.
lo, hi = 0.20, 0.35
n_s = len(speed)
# finished = n_s / frac  => n_d = finished - n_s
# want n_s / (n_d + n_s) in [0.20, 0.35]
# n_d <= n_s / 0.20 - n_s = 4 n_s
# n_d >= n_s / 0.35 - n_s ≈ 1.857 n_s
max_d = int(n_s / lo - n_s)
min_d = int(n_s / hi - n_s)
n_d = len(distill)
if n_d > max_d:
    by = {}
    for r in distill:
        by.setdefault((r["level"], r["problem_id"]), []).append(r)
    keys = sorted(by)
    random.Random(0).shuffle(keys)
    picked, n = [], 0
    # Round-robin so a cap still covers many tasks.
    i = 0
    while n < max_d:
        progressed = False
        for k in keys:
            if n >= max_d:
                break
            if i < len(by[k]):
                picked.append(by[k][i])
                n += 1
                progressed = True
        if not progressed:
            break
        i += 1
    distill = picked
    n_d = len(distill)
frac = n_s / max(n_d + n_s, 1)
print("mix: distill %d  speed %d  speed_frac %.3f (target %.2f-%.2f)"
      % (n_d, n_s, frac, lo, hi))
if frac < lo - 1e-9:
    raise SystemExit("speed slice drowned; refuse to write")
if frac > hi + 1e-9 and n_d < min_d:
    print("speed_frac above 0.35 because distill is small; keeping all distill")
rows = distill + speed
random.Random(1).shuffle(rows)
with open(sys.argv[3], "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote %d rows %s" % (len(rows), sys.argv[3]))
PY
