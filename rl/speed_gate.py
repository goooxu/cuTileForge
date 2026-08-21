#!/usr/bin/env python3
"""Decide whether a timed harvest has a catchable-speed slice.

Gate (plan):
  - family is not standalone matmul / conv
  - 0.40 <= speedup < 1.05
  - the problem has at least one passing timed sample
  - at least 80 problems or 200 timed passing samples
  - matmul+conv must be <= 30% of the slice (should be ~0 if builders are right)

Exit 0 if the gate passes, 1 if it does not. Always prints the distribution.
"""
import argparse
import collections
import json
import os
import re
import sys


BLOCK = {"matmul", "conv"}


def category_of(src):
    m = re.search(r'"""(\w+) \(tier (\d+), (\w+)\)', src)
    return m.group(3) if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--min-speedup", type=float, default=0.40)
    ap.add_argument("--max-speedup", type=float, default=1.05)
    ap.add_argument("--min-problems", type=int, default=80)
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--max-block-frac", type=float, default=0.30)
    ap.add_argument("--root", default=None,
                    help="KernelBench root. Default: repo kernelbench/KernelBench.")
    args = ap.parse_args()

    forge = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.root or os.path.join(forge, "kernelbench", "KernelBench")
    lvdir = os.path.join(root, "level%d" % args.level)
    cats = {}
    for fn in os.listdir(lvdir):
        if not fn.endswith(".py"):
            continue
        pid = int(fn.split("_", 1)[0])
        cats[pid] = category_of(open(os.path.join(lvdir, fn),
                                     encoding="utf-8", errors="replace").read())

    n_pass = n_timed = 0
    by_cat = collections.Counter()
    slice_cat = collections.Counter()
    slice_problems = set()
    slice_n = 0
    all_su = []
    slice_su = []
    for line in open(args.verified):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("passed"):
            continue
        n_pass += 1
        pid = int(r["key"].split(":")[0])
        cat = cats.get(pid, "?")
        by_cat[cat] += 1
        su = r.get("speedup")
        if su is None:
            continue
        n_timed += 1
        all_su.append(su)
        if args.min_speedup <= su < args.max_speedup:
            slice_n += 1
            slice_su.append(su)
            slice_problems.add(pid)
            slice_cat[cat] += 1

    def pct(xs, pred):
        return 100.0 * sum(1 for x in xs if pred(x)) / max(len(xs), 1)

    print("passed %d  timed %d  catchable samples %d  catchable problems %d"
          % (n_pass, n_timed, slice_n, len(slice_problems)))
    print("passed by family: %s" % dict(by_cat))
    print("catchable by family: %s" % dict(slice_cat))
    if all_su:
        s = sorted(all_su)
        print("all timed speedup: min %.3f  median %.3f  max %.3f  "
              "<0.40 %0.1f%%  [0.40,1.05) %0.1f%%  >=1.05 %0.1f%%"
              % (s[0], s[len(s) // 2], s[-1],
                 pct(s, lambda x: x < 0.40),
                 pct(s, lambda x: 0.40 <= x < 1.05),
                 pct(s, lambda x: x >= 1.05)))
    if slice_su:
        s = sorted(slice_su)
        print("catchable speedup: min %.3f  median %.3f  max %.3f"
              % (s[0], s[len(s) // 2], s[-1]))

    block = sum(slice_cat[c] for c in BLOCK)
    block_frac = block / max(slice_n, 1)
    print("matmul+conv in slice: %d/%d (%.1f%%)" % (block, slice_n, 100 * block_frac))

    ok_n = len(slice_problems) >= args.min_problems or slice_n >= args.min_samples
    ok_mix = block_frac <= args.max_block_frac
    if ok_n and ok_mix and slice_n > 0:
        print("GATE: pass")
        return 0
    print("GATE: fail")
    if not slice_n:
        print("  empty catchable set")
    if not ok_n:
        print("  need >= %d problems or >= %d samples" %
              (args.min_problems, args.min_samples))
    if not ok_mix:
        print("  matmul/conv share %.1f%% > %.0f%%" %
              (100 * block_frac, 100 * args.max_block_frac))
    return 1


if __name__ == "__main__":
    sys.exit(main())
