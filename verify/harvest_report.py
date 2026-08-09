#!/usr/bin/env python3
"""Summarise a verified harvest: coverage, reliability, and the gap between them.

The point of sampling a task many times is to separate "the model cannot do
this" from "the model can do this but rarely". Only the second group is worth
training on for reliability, and it is invisible in a pass@1 number.

  python3 verify/harvest_report.py --verified runs/harvest_k16/verified.jsonl
"""
import argparse
import collections
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    by_task = collections.defaultdict(list)
    stages = collections.Counter()
    for line in open(args.verified):
        rec = json.loads(line)
        pid = int(rec["key"].split(":")[0])
        by_task[pid].append(rec)
        stages[rec.get("stage") or ("pass" if rec.get("passed") else "?")] += 1

    n_recs = sum(len(v) for v in by_task.values())
    n_pass = sum(1 for v in by_task.values() for r in v if r.get("passed"))
    solved = {p: sum(1 for r in v if r.get("passed")) for p, v in by_task.items()}

    print("%d tasks, %d verified samples" % (len(by_task), n_recs))
    print("  per-sample pass rate      %.1f%%" % (100.0 * n_pass / max(n_recs, 1)))
    print("  tasks solved at least once %d (%.1f%%)"
          % (sum(1 for n in solved.values() if n),
             100.0 * sum(1 for n in solved.values() if n) / len(by_task)))

    # The three groups that matter. Only the middle one can teach reliability:
    # a task never solved has nothing to imitate, and one always solved has
    # nothing left to gain.
    never = sum(1 for n in solved.values() if n == 0)
    always = sum(1 for p, n in solved.items() if n == len(by_task[p]))
    frontier = len(by_task) - never - always
    print()
    print("  never solved     %4d  (needs new capability, not sharpening)" % never)
    print("  sometimes solved %4d  (the reliability frontier)" % frontier)
    print("  always solved    %4d  (nothing left to gain)" % always)

    print()
    print("  distribution of successes per task")
    hist = collections.Counter(solved.values())
    for n in sorted(hist):
        print("    %2d/%d  %4d tasks" % (n, args.k, hist[n]))

    timed = [r for v in by_task.values() for r in v if r.get("speedup")]
    if timed:
        sp = sorted(r["speedup"] for r in timed)
        print()
        print("  timed %d correct kernels: median %.3fx, %d beat torch (%.1f%%)"
              % (len(sp), sp[len(sp) // 2], sum(1 for s in sp if s > 1.0),
                 100.0 * sum(1 for s in sp if s > 1.0) / len(sp)))

    print()
    print("  failure stages: %s" % dict(stages.most_common(8)))


if __name__ == "__main__":
    main()
