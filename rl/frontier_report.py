#!/usr/bin/env python3
"""Category composition of a frontier, next to the benchmark it is meant to move.

Selecting the frontier purely by reward spread lets whichever family happens to
be most numerous take most of the rollouts. Stacking GRPO on the self-distilled
model gained 19 problems overall but lost 3 on matmul and 2 on activation, and
left pooling at 1 of 10 -- so what the frontier is made of is worth looking at
directly rather than inferring from the result.

  python3 rl/frontier_report.py --frontier runs/rl_frontier_H.json
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "train"))

from build_sft_dataset import category_of  # noqa: E402

# What the 200-problem dev set is made of, for comparison.
DEV_MIX = {"conv": 98, "activation": 29, "norm": 24, "matmul": 15,
           "reduction": 11, "pool": 10, "other": 7, "loss": 6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--top", type=int, default=None,
                    help="Only count the first N entries, i.e. what a run that "
                         "samples from the head of the sorted list would see.")
    args = ap.parse_args()

    entries = json.load(open(args.frontier))
    if args.top:
        entries = entries[:args.top]

    cats = collections.Counter()
    spread = collections.defaultdict(list)
    rate = collections.defaultdict(list)
    for e in entries:
        c = category_of(e.get("ref_src", "")) if e.get("ref_src") else "?"
        if c == "?":
            # Fall back to the task name, which the generator builds from the
            # operator it used.
            name = e.get("problem", "")
            m = re.search(r"(conv|pool|norm|matmul|reduc|softmax|chain)", name, re.I)
            c = m.group(1).lower() if m else "?"
        cats[c] += 1
        spread[c].append(e.get("reward_spread", 0.0))
        rate[c].append(e.get("pass_rate", 0.0))

    n = sum(cats.values())
    dev_total = sum(DEV_MIX.values())
    print("%s: %d tasks" % (args.frontier, n))
    print()
    print("  category        n   share   dev share   median spread   median rate")
    for c, k in cats.most_common():
        s = sorted(spread[c])
        r = sorted(rate[c])
        dev = DEV_MIX.get(c)
        dev_s = "%5.1f%%" % (100.0 * dev / dev_total) if dev else "    --"
        print("  %-12s %4d  %5.1f%%     %s      %6.3f        %6.3f"
              % (c, k, 100.0 * k / n, dev_s, s[len(s) // 2], r[len(r) // 2]))


if __name__ == "__main__":
    main()
