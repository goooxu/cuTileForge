#!/usr/bin/env python3
"""Hard gates on a timed harvest before a speed RL run is allowed to start.

The last speed round trained anyway and discovered afterwards that the tasks
were in the wrong regime. These two checks are the same questions, asked
before any weights move:

  1. Speed band: median best_speedup in [0.4, 1.0] against torch.compile, and
     at least half the timed tasks inside (0.4, 1.0). Outside that, either the
     model already wins (nothing to learn) or the bonus is clamped (flat).
  2. Within-task variance: at least a third of tasks with >=3 timed passes
     have best/median >= 1.3x -- the same bar as speed_probe.py. No variance
     means the gap is knowledge, and GRPO cannot close it.

  python3 verify/speed_gates.py --verified runs/harvest_bw83_verified.jsonl
"""
import argparse
import collections
import json
import statistics
import sys

BAND_LO, BAND_HI = 0.4, 1.0
BAND_SHARE = 0.5
VAR_RATIO = 1.3
VAR_SHARE = 1.0 / 3.0
MIN_TIMED = 3


def load(path):
    by = collections.defaultdict(list)
    for line in open(path):
        rec = json.loads(line)
        pid = int(rec["key"].split(":")[0])
        by[pid].append(rec)
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True)
    args = ap.parse_args()

    by = load(args.verified)
    bests, ratios = [], []
    for recs in by.values():
        sp = [r["speedup"] for r in recs if r.get("passed") and r.get("speedup")]
        if not sp:
            continue
        bests.append(max(sp))
        if len(sp) >= MIN_TIMED:
            med = statistics.median(sp)
            ratios.append(max(sp) / med if med else 0.0)

    fails = []
    print("timed tasks with at least one pass: %d" % len(bests))
    if not bests:
        print("GATE 1 FAIL: no timed passes")
        raise SystemExit(1)

    bests.sort()
    med = bests[len(bests) // 2]
    in_band = sum(1 for s in bests if BAND_LO < s < BAND_HI)
    share = in_band / float(len(bests))
    print("  best_speedup  min %.3f  median %.3f  max %.3f"
          % (bests[0], med, bests[-1]))
    print("  in (%.2f, %.2f): %d/%d (%.0f%%)"
          % (BAND_LO, BAND_HI, in_band, len(bests), 100 * share))
    if not (BAND_LO <= med <= BAND_HI):
        fails.append("median best_speedup %.3f is outside [%.2f, %.2f]"
                     % (med, BAND_LO, BAND_HI))
    if share < BAND_SHARE:
        fails.append("only %.0f%% of tasks sit in the band (need %.0f%%)"
                     % (100 * share, 100 * BAND_SHARE))
    print("  GATE 1: %s" % ("PASS" if not fails else "FAIL -- " + fails[-1]))

    print()
    print("tasks with >=%d timed passes: %d" % (MIN_TIMED, len(ratios)))
    if not ratios:
        fails.append("no task has %d timed passes; cannot judge variance"
                     % MIN_TIMED)
        print("  GATE 2 FAIL: %s" % fails[-1])
    else:
        n_spread = sum(1 for r in ratios if r >= VAR_RATIO)
        vshare = n_spread / float(len(ratios))
        print("  median best/median: %.2fx" % statistics.median(ratios))
        print("  %d/%d at or above %.1fx (%.0f%%)"
              % (n_spread, len(ratios), VAR_RATIO, 100 * vshare))
        if vshare < VAR_SHARE:
            fails.append("variance share %.0f%% is below %.0f%%"
                         % (100 * vshare, 100 * VAR_SHARE))
            print("  GATE 2 FAIL: %s" % fails[-1])
        else:
            print("  GATE 2: PASS")

    print()
    if fails:
        print("STOP -- do not train. " + "; ".join(fails))
        raise SystemExit(1)
    print("both gates passed; the harvest is in the regime speed RL can use")


if __name__ == "__main__":
    main()
