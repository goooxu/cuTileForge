#!/usr/bin/env python3
"""Combine frontiers screened from several task pools, then cap each family.

A frontier is screened one level at a time because problem ids only mean
something within a level. Covering the benchmark's operator mix takes more than
one pool, though: the original synthetic ladder has no activation or loss family
at all, and those are 35 of the 200 benchmark problems. So screen each pool
separately and join them here.

The cap matters as much as the join. Left to sort by reward spread, the frontier
takes whatever shape the pool had -- 65% convolution and pooling in the run that
stacked GRPO on the distilled model, which gained 19 problems overall but lost 3
on matmul and 2 on activation.

  python3 rl/merge_frontier.py --in runs/rl_frontier_H.json \\
      --in runs/rl_frontier_w.json --quota conv=200,pool=120,... \\
      --out runs/rl_frontier_bal.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from select_frontier import apply_category_quota  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True)
    ap.add_argument("--quota", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    merged, seen = [], set()
    for path in args.inputs:
        entries = json.load(open(path))
        n_new = 0
        for e in entries:
            # (level, problem_id) is the only identifier that survives across
            # pools; ids restart at 1 in every level.
            key = (e["level"], e["problem_id"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(e)
            n_new += 1
        print("%s: %d entries, %d new" % (path, len(entries), n_new))

    # Interleave by spread so the cap trims each pool's weakest tasks rather
    # than whichever file happened to come second.
    merged.sort(key=lambda e: (-e.get("reward_spread", 0.0),
                               -e.get("pass_rate", 0.0)))

    if args.quota:
        merged = apply_category_quota(merged, args.quota)

    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)

    cats = collections.Counter(e.get("category", "?") for e in merged)
    lvls = collections.Counter(e["level"] for e in merged)
    print("wrote %d tasks to %s" % (len(merged), args.out))
    print("  by category: %s" % dict(cats.most_common()))
    print("  by level:    %s" % dict(sorted(lvls.items())))


if __name__ == "__main__":
    main()
