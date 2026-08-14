#!/usr/bin/env python3
"""Per-problem speed ratio on the problems two runs both solved.

Comparing median speedup across different solved-sets is a composition effect.
The speed round that looked like 43 -> 46 / 0.92x -> 0.94x was 1.000x on the
136 problems both models actually solved. This is that check.

  python3 verify/pairwise_speed.py \\
      --a M:runs/M_k4_l1,runs/M_k4_l2 --b S:runs/S_k16_l1,runs/S_k16_l2 \\
      --analysis-name analysis_compile.json
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "train"))
from compare_partial import categorise  # noqa: E402

ANCHORS = ("27_SELU_", "31_ELU", "32_HardTanh")


def load_best(spec, analysis_name):
    """spec is LABEL:DIR[,DIR...] -- directories of one run, any number of levels."""
    label, _, paths = spec.rpartition(":")
    best, names = {}, {}
    for d in paths.split(","):
        recs = json.load(open(os.path.join(d, analysis_name)))
        if isinstance(recs, dict) and "records" in recs:
            recs = recs["records"]
        for r in recs:
            if not (r.get("passed") and r.get("fully_cutile") and r.get("speedup")):
                continue
            # Problem identity is (problem name): ids restart per level.
            key = r.get("problem") or ("%s/%s" % (d, r["problem_id"]))
            best[key] = max(best.get(key, 0.0), r["speedup"])
            names[key] = r.get("problem") or key
    return label or paths, best, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, metavar="LABEL:DIR[,DIR...]")
    ap.add_argument("--b", required=True, metavar="LABEL:DIR[,DIR...]")
    ap.add_argument("--analysis-name", default="analysis_compile.json")
    ap.add_argument("--anchors", default=",".join(ANCHORS),
                    help="Comma-separated problem-name prefixes to call out.")
    args = ap.parse_args()

    la, a, na = load_best(args.a, args.analysis_name)
    lb, b, nb = load_best(args.b, args.analysis_name)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no commonly solved timed problems")

    ratios = [b[k] / a[k] for k in common if a[k] > 0]
    faster = sum(1 for r in ratios if r > 1.05)
    slower = sum(1 for r in ratios if r < 1 / 1.05)
    tight = sum(1 for r in ratios if 1 / 1.05 <= r <= 1.05)
    med = statistics.median(ratios)
    print("%s vs %s on %d commonly solved timed problems"
          % (lb, la, len(common)))
    print("  median %s/%s = %.3fx" % (lb, la, med))
    print("  %s faster (>1.05x) %d   slower %d   within ±5%% %d"
          % (lb, faster, slower, tight))

    print()
    print("  by family (median ratio, n):")
    by = {}
    for k in common:
        by.setdefault(categorise(na.get(k, k)), []).append(b[k] / a[k])
    for cat, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print("    %-12s n=%3d  median %.3fx" % (cat, len(rs),
                                                 statistics.median(rs)))

    print()
    print("  anchors:")
    prefixes = [p.strip() for p in args.anchors.split(",") if p.strip()]
    for prefix in prefixes:
        hits = [k for k in common if prefix.lower() in k.lower()]
        if not hits:
            # Still report if only one side solved it.
            only_a = [k for k in a if prefix.lower() in k.lower()]
            only_b = [k for k in b if prefix.lower() in k.lower()]
            print("    %-20s not in the common set  (a=%s b=%s)"
                  % (prefix, ",".join("%.3fx" % a[k] for k in only_a) or "-",
                     ",".join("%.3fx" % b[k] for k in only_b) or "-"))
            continue
        for k in hits:
            print("    %-40s  %s %.3fx  %s %.3fx  ratio %.3fx"
                  % (k[:40], la, a[k], lb, b[k], b[k] / a[k]))


if __name__ == "__main__":
    main()
