#!/usr/bin/env python3
"""Category composition of one or more task levels.

Written because the first sealed held-out turned out not to cover the families
that later rounds worked on: levels 97 and 98 predate the activation and loss
builders, so they contain no standalone task from either, and could not speak to
those rounds at all. Checking the mix before sealing is cheaper than finding out
afterwards.

  python3 taskgen/level_categories.py --levels 99,88,97,98
"""
import argparse
import collections
import os
import re

DOCSTRING = re.compile(r'"""(\w+) \(tier (\d+), (\w+)\)')


def categories(level_dir):
    counts = collections.Counter()
    for fname in sorted(os.listdir(level_dir)):
        if not fname.endswith(".py"):
            continue
        src = open(os.path.join(level_dir, fname), encoding="utf-8",
                   errors="replace").read()
        m = DOCSTRING.search(src)
        counts[m.group(3) if m else "?"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kernelbench/KernelBench")
    ap.add_argument("--levels", required=True)
    args = ap.parse_args()

    for lvl in [int(x) for x in args.levels.split(",")]:
        d = os.path.join(args.root, "level%d" % lvl)
        if not os.path.isdir(d):
            print("level%-3d (missing)" % lvl)
            continue
        c = categories(d)
        total = sum(c.values())
        parts = ", ".join("%s %d" % (k, v) for k, v in c.most_common())
        print("level%-3d %4d tasks: %s" % (lvl, total, parts))


if __name__ == "__main__":
    main()
