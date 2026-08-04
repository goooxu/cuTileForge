"""Where a task set's kernels win on speed, and where they cannot.

A single median speedup hides the only thing worth knowing. On the fusion task
set the aggregate was 0.082x, which reads as a total failure, while underneath it
long elementwise chains beat torch 98% of the time at up to 3.98x and everything
touching matmul or conv lost 100% of the time. Those two facts point at opposite
conclusions and only the breakdown shows both.

Groups by operator pattern rather than by the coarse category, since
MatmulBiasReLU and MatmulBiasTanh are one pattern with two tails.

Usage:
    python3 verify/speed_report.py --verified runs/repair_l94_verified.jsonl \\
        --level 94
"""

import argparse
import collections
import json
import os
import re
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True,
                    help="JSONL from fast_verify.py --measure-time.")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--repo-root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    args = ap.parse_args()

    prob_dir = os.path.join(args.repo_root, "kernelbench", "KernelBench",
                            "level%d" % args.level)
    names = {}
    for f in os.listdir(prob_dir):
        m = re.match(r"(\d+)_", f)
        d = re.search(r'"""(\w+) \(tier \d+, (\w+)\)',
                      open(os.path.join(prob_dir, f)).read())
        if m and d:
            names[int(m.group(1))] = d.group(1)

    # The activation tail is a variation on a pattern, not a pattern of its own.
    tail = re.compile(r"(ReLU|Sigmoid|Tanh|Scale|AddBias)$")

    by_pattern = collections.defaultdict(list)
    n_untimed = 0
    for line in open(args.verified):
        r = json.loads(line)
        if not r.get("passed"):
            continue
        if not r.get("speedup"):
            n_untimed += 1
            continue
        name = names.get(int(r["key"].split(":")[0]), "?")
        by_pattern[tail.sub("", name) or name].append(r["speedup"])

    if not by_pattern:
        raise SystemExit("no timed passes; was --measure-time used?")

    print("%-22s %6s %9s %9s %12s" %
          ("fusion pattern", "n", "median", "max", "beat torch"))
    for pat in sorted(by_pattern, key=lambda p: -len(by_pattern[p])):
        sp = by_pattern[pat]
        fast = sum(1 for s in sp if s > 1.0)
        print("%-22s %6d %8.3fx %8.2fx %6d (%3.0f%%)"
              % (pat, len(sp), statistics.median(sp), max(sp), fast,
                 fast / len(sp) * 100))

    allsp = [s for v in by_pattern.values() for s in v]
    fast = sum(1 for s in allsp if s > 1.0)
    print("\n%d timed, %d beat torch (%.0f%%), median %.3fx"
          % (len(allsp), fast, fast / len(allsp) * 100, statistics.median(allsp)))
    if n_untimed:
        print("%d correct but untimed" % n_untimed)

    winners = [p for p, sp in by_pattern.items()
               if sum(1 for s in sp if s > 1.0) / len(sp) > 0.5]
    losers = [p for p, sp in by_pattern.items()
              if not any(s > 1.0 for s in sp)]
    if winners:
        print("wins on: %s" % ", ".join(sorted(winners)))
    if losers:
        print("never wins on: %s" % ", ".join(sorted(losers)))


if __name__ == "__main__":
    main()
