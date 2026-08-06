#!/usr/bin/env python3
"""pass@k as a function of k for one run, from its analysis.json.

Exists to answer one question that the headline metrics cannot: when a
multi-attempt method beats a single-shot run, is the gain from the feedback or
just from having taken more shots? Comparing an n-round repair loop against a
single-shot run at the same total sample budget needs the single-shot pass@k
curve, not only pass@1 and pass@4.

  python3 scripts/passk_curve.py --run ../runs/F_k16_l2 --level 2
"""
import argparse
import collections
import json
import os


def load(run_dir):
    recs = json.load(open(os.path.join(run_dir, "analysis.json")))
    if isinstance(recs, dict) and "records" in recs:
        recs = recs["records"]
    return recs


def passed(r):
    """The project's criterion: numerically correct AND entirely cuTile."""
    return bool(r.get("passed")) and bool(r.get("fully_cutile"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--ks", default="1,2,4,8,16")
    args = ap.parse_args()

    recs = load(args.run)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["problem_id"]].append(r)
    n_probs = len(by)

    unevaluated = sum(1 for r in recs if not r.get("evaluated"))
    if unevaluated:
        print("WARNING: %d of %d samples were never evaluated; pass@k is a "
              "lower bound and the run should be resumed before it is used as a "
              "control" % (unevaluated, len(recs)))

    print("%s: %d problems, %d samples" % (args.run, n_probs, len(recs)))
    print()
    print("  k     solved      pass@k")
    for k in [int(x) for x in args.ks.split(",")]:
        n = 0
        for rs in by.values():
            head = sorted(rs, key=lambda x: x["sample_id"])[:k]
            if any(passed(x) for x in head):
                n += 1
        print("  %-4d  %3d/%-3d     %5.1f%%" % (k, n, n_probs,
                                                100.0 * n / n_probs))

    per = 100.0 * sum(1 for r in recs if passed(r)) / len(recs)
    fast = sum(1 for rs in by.values()
               if any(passed(x) and (x.get("speedup") or 0) >= 1.0 for x in rs))
    print()
    print("  per-sample pass rate   %.1f%%" % per)
    print("  fast_1.0 at full k     %d/%d" % (fast, n_probs))


if __name__ == "__main__":
    main()
