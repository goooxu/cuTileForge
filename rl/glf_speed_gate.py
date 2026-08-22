#!/usr/bin/env python3
"""Hard gates on the GL-F timed harvest before any weights move.

Uses the same two checks as verify/speed_gates.py on the union of the
GL-E harvests, plus a conv/matmul share on the slow-but-solid slice
(best < 1.0). A pointwise-only pool cannot transfer to table A latency.

  python3 rl/glf_speed_gate.py \\
      --run 86:runs/harvest_gle86_verified.jsonl \\
      --run 87:runs/harvest_gle87_verified.jsonl \\
      --run 92:runs/harvest_gle92_verified.jsonl \\
      --run 93:runs/harvest_gle93_verified.jsonl
"""
from __future__ import print_function

import argparse
import collections
import json
import os
import re
import statistics
import sys

BAND_LO, BAND_HI = 0.4, 1.0
BAND_SHARE = 0.5
VAR_RATIO = 1.3
VAR_SHARE = 1.0 / 3.0
MIN_TIMED = 3
GEMM_SHARE = 0.30
BLOCK = ("matmul", "conv")


def category_of(src):
    m = re.search(r'"""(\w+) \(tier (\d+), (\w+)\)', src)
    return m.group(3) if m else "?"


def load_cats(root, level):
    lvdir = os.path.join(root, "level%d" % level)
    cats = {}
    if not os.path.isdir(lvdir):
        return cats
    for fn in os.listdir(lvdir):
        if not fn.endswith(".py"):
            continue
        pid = int(fn.split("_", 1)[0])
        cats[pid] = category_of(open(os.path.join(lvdir, fn),
                                    encoding="utf-8", errors="replace").read())
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="LEVEL:VERIFIED")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    forge = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.root or os.path.join(forge, "kernelbench", "KernelBench")

    by = collections.defaultdict(list)
    cats = {}
    for spec in args.run:
        level_s, verified = spec.split(":", 1)
        level = int(level_s)
        level_cats = load_cats(root, level)
        for line in open(verified):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("passed") or rec.get("speedup") is None:
                continue
            pid = int(rec["key"].split(":")[0])
            key = (level, pid)
            by[key].append(rec["speedup"])
            cats[key] = level_cats.get(pid, "?")

    bests, ratios = [], []
    slow_keys = []
    for key, sp in by.items():
        bests.append((key, max(sp)))
        if max(sp) < 1.0:
            slow_keys.append(key)
        if len(sp) >= MIN_TIMED:
            med = statistics.median(sp)
            ratios.append(max(sp) / med if med else 0.0)

    fails = []
    print("timed tasks with at least one pass: %d" % len(bests))
    if not bests:
        print("GATE 1 FAIL: no timed passes")
        raise SystemExit(1)

    best_vals = sorted(b for _, b in bests)
    med = best_vals[len(best_vals) // 2]
    in_band = sum(1 for s in best_vals if BAND_LO < s < BAND_HI)
    share = in_band / float(len(best_vals))
    print("  best_speedup  min %.3f  median %.3f  max %.3f"
          % (best_vals[0], med, best_vals[-1]))
    print("  in (%.2f, %.2f): %d/%d (%.0f%%)"
          % (BAND_LO, BAND_HI, in_band, len(best_vals), 100 * share))
    gate1 = []
    if not (BAND_LO <= med <= BAND_HI):
        gate1.append("median best_speedup %.3f is outside [%.2f, %.2f]"
                     % (med, BAND_LO, BAND_HI))
    if share < BAND_SHARE:
        gate1.append("only %.0f%% of tasks sit in the band (need %.0f%%)"
                     % (100 * share, 100 * BAND_SHARE))
    fails.extend(gate1)
    print("  GATE 1: %s" % ("PASS" if not gate1 else "FAIL -- " + gate1[-1]))

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
    slow_cat = collections.Counter(cats.get(k, "?") for k in slow_keys)
    gemm = sum(slow_cat[c] for c in BLOCK)
    gemm_frac = gemm / float(max(len(slow_keys), 1))
    print("slow-but-solid (best < 1.0): %d" % len(slow_keys))
    print("  by family: %s" % dict(slow_cat))
    print("  conv+matmul: %d/%d (%.1f%%)" % (gemm, len(slow_keys), 100 * gemm_frac))
    if len(slow_keys) == 0 or gemm_frac < GEMM_SHARE:
        fails.append("conv+matmul share %.1f%% of slow-but-solid is below %.0f%%"
                     % (100 * gemm_frac, 100 * GEMM_SHARE))
        print("  GATE 3 FAIL: %s" % fails[-1])
    else:
        print("  GATE 3: PASS")

    print()
    if fails:
        print("STOP -- do not train. " + "; ".join(fails))
        raise SystemExit(1)
    print("all gates passed; the harvest is in the regime GL-F can use")


if __name__ == "__main__":
    main()
