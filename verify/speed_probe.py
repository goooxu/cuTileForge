#!/usr/bin/env python3
"""Does the model produce kernels of differing speed for a problem it has mastered?

This decides whether speed is an optimisation problem or a knowledge one, and the
answer changes what is worth doing next. Group-relative advantage can only exploit
variance that exists: if every sample for a problem compiles to the same kernel at
the same speed, no amount of reinforcement learning will find a faster one, and
the gap is that nothing ever taught the model what makes a cuTile kernel fast --
the concepts tier says nothing about it and the API reference barely more.

At the sampling temperature used for evaluation the spread is already known to be
small: of the problems the best model solves 4 out of 4, only 3 of 39 on Level 1
and 0 of 15 on Level 2 show even a 1.3x difference between their own samples. This
turns the temperature up and takes many more samples, to separate "there is no
variance to find" from "we were not looking hard enough".

The verdict is fixed before the run to keep it honest:

  at least a third of problems show best/median >= 1.3x
      -> variance exists, speed is an exploration problem, RL can reach it
  nearly all problems stay under 1.2x
      -> one implementation per problem, speed is a knowledge gap, RL cannot

Tile-size literals are extracted as a second, more direct read on the same thing:
if those never move, the model is not even touching the most basic performance
knob.

  python3 verify/speed_probe.py --run runs/speed_probe_l1 --level 1
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VERDICT_SPREAD = 1.3
VERDICT_SHARE = 1.0 / 3.0
TIGHT = 1.2


def tile_literals(code: str):
    """Tile-shape constants, as a proxy for the choices that set performance.

    cuTile needs these as compile-time constants, so they appear as module-level
    integer literals or inline in load()/num_tiles() calls. Exact recall is not
    the point; what matters is whether they vary between samples at all.
    """
    lits = set()
    for m in re.finditer(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)\s*$", code,
                         re.MULTILINE):
        if any(t in m.group(1) for t in ("TILE", "BLOCK", "BM", "BN", "BK",
                                         "CHUNK", "SIZE")):
            lits.add("%s=%s" % (m.group(1), m.group(2)))
    return frozenset(lits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True,
                    help="JSONL from fast_verify.py --measure-time.")
    ap.add_argument("--kernel-dir", default=None,
                    help="Generated kernels, for the tile-size read.")
    ap.add_argument("--level", type=int, default=None)
    args = ap.parse_args()

    by_task = collections.defaultdict(list)
    for line in open(args.verified):
        rec = json.loads(line)
        by_task[int(rec["key"].split(":")[0])].append(rec)

    rows = []
    for pid, recs in sorted(by_task.items()):
        sp = [r["speedup"] for r in recs
              if r.get("passed") and r.get("speedup")]
        if len(sp) < 3:
            continue
        sp.sort()
        med = statistics.median(sp)
        rows.append((pid, len(sp), med, max(sp), max(sp) / med if med else 0.0))

    if not rows:
        print("no problem had enough correct timed samples to judge")
        return

    print("problems with >=3 correct timed samples: %d" % len(rows))
    print()
    print("  problem   n   median    best   best/median")
    for pid, n, med, best, ratio in rows:
        print("  %-8d %3d  %6.3fx %6.3fx    %5.2fx" % (pid, n, med, best, ratio))

    ratios = [r[4] for r in rows]
    n_spread = sum(1 for r in ratios if r >= VERDICT_SPREAD)
    n_tight = sum(1 for r in ratios if r < TIGHT)
    share = n_spread / float(len(ratios))

    print()
    print("  median best/median across problems: %.2fx" % statistics.median(ratios))
    print("  %d/%d problems at or above %.1fx (%.0f%%)"
          % (n_spread, len(ratios), VERDICT_SPREAD, 100 * share))
    print("  %d/%d problems below %.1fx" % (n_tight, len(ratios), TIGHT))
    print()
    if share >= VERDICT_SHARE:
        print("  VERDICT: variance exists -- speed is an exploration problem, "
              "and RL on a speed-only objective can reach it")
    elif n_tight >= 0.9 * len(ratios):
        print("  VERDICT: one implementation per problem -- speed is a knowledge "
              "gap, and more RL will not close it")
    else:
        print("  VERDICT: inconclusive; between the two thresholds")

    if args.kernel_dir and args.level is not None:
        print()
        pattern = re.compile(
            r"level_%d_problem_(\d+)_sample_(\d+)_kernel\.py" % args.level)
        per_task = collections.defaultdict(set)
        for fname in os.listdir(args.kernel_dir):
            m = pattern.match(fname)
            if not m:
                continue
            code = open(os.path.join(args.kernel_dir, fname),
                        encoding="utf-8", errors="replace").read()
            per_task[int(m.group(1))].add(tile_literals(code))
        varied = sum(1 for v in per_task.values() if len(v) > 1)
        print("  tile-size choices: %d/%d problems used more than one set"
              % (varied, len(per_task)))
        if per_task:
            ex = sorted(per_task.items())[0]
            print("  example problem %d used %d distinct set(s): %s"
                  % (ex[0], len(ex[1]),
                     [sorted(s)[:3] for s in list(ex[1])[:3]]))


if __name__ == "__main__":
    main()
