"""Why convolution stays at zero.

The repair loop lifted every other category but left conv at 0 of 480
candidates over four attempts each. That is too systematic to be difficulty
alone, so this groups conv failures by their actual cuTile diagnostic to see
whether one blocker accounts for them.

Usage:
    python3 repair/conv_postmortem.py --run runs/repair_l93 --level 93
"""

import argparse
import collections
import json
import os
import re

import analyze_repair as ar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--level", type=int, default=93)
    ap.add_argument("--show", type=int, default=6, help="Example errors per class.")
    args = ap.parse_args()

    prob_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "kernelbench", "KernelBench",
                            "level%d" % args.level)
    cat_of = {}
    for f in os.listdir(prob_dir):
        m = re.match(r"(\d+)_", f)
        if not m:
            continue
        d = re.search(r'"""(\w+) \(tier (\d+), (\w+)\)',
                      open(os.path.join(prob_dir, f)).read())
        if d:
            cat_of[int(m.group(1))] = (d.group(1), int(d.group(2)), d.group(3))

    trajs = [json.loads(l) for l in open(os.path.join(args.run, "trajectories.jsonl"))]
    conv = [t for t in trajs if cat_of.get(t["problem_id"], ("", 0, ""))[2] == "conv"]

    print("conv candidates: %d over %d problems"
          % (len(conv), len({t["problem_id"] for t in conv})))

    # Final-attempt failure class, i.e. where each candidate gave up.
    final = collections.Counter()
    examples = collections.defaultdict(list)
    for t in conv:
        last = t["history"][-1]
        cls = ar.classify(last["stage"], last["error"])
        final[cls] += 1
        if len(examples[cls]) < args.show:
            examples[cls].append((cat_of[t["problem_id"]][0], last["error"][:230]))

    print("\n### where conv candidates end up (final attempt)\n")
    n = len(conv)
    for cls, c in final.most_common():
        print("  %-22s %5d  %5.1f%%" % (cls, c, c / n * 100))

    print("\n### representative errors\n")
    for cls, _ in final.most_common(5):
        print("  --- %s ---" % cls)
        for op, err in examples[cls]:
            print("    [%s] %s" % (op, " ".join(err.split())[:200]))
        print()

    # Does the model at least get closer? Compare first vs last attempt class.
    moved = collections.Counter()
    for t in conv:
        h = t["history"]
        a = ar.classify(h[0]["stage"], h[0]["error"])
        b = ar.classify(h[-1]["stage"], h[-1]["error"])
        moved[(a, b)] += 1
    print("### most common first-attempt -> final-attempt transitions\n")
    for (a, b), c in moved.most_common(8):
        arrow = "stayed" if a == b else "became"
        print("  %-20s %s %-20s %4d" % (a, arrow, b, c))

    # By tier: is even the easiest conv out of reach?
    print("\n### conv by tier (all failed; shows where they die)\n")
    bytier = collections.defaultdict(collections.Counter)
    for t in conv:
        tier = cat_of[t["problem_id"]][1]
        bytier[tier][ar.classify(t["history"][-1]["stage"],
                                 t["history"][-1]["error"])] += 1
    for tier in sorted(bytier):
        top = ", ".join("%s:%d" % (k, v) for k, v in bytier[tier].most_common(3))
        print("  tier %d (%3d cands): %s" % (tier, sum(bytier[tier].values()), top))

    # numeric_mismatch means it compiled and ran -- how close was it?
    print("\n### conv candidates that compiled but computed wrong values\n")
    diffs = []
    for t in conv:
        for h in t["history"]:
            m = re.search(r"max diff ([0-9.eE+-]+)", h["error"] or "")
            if m:
                try:
                    diffs.append(float(m.group(1)))
                except ValueError:
                    pass
    if diffs:
        diffs.sort()
        print("  attempts reaching a numeric comparison: %d" % len(diffs))
        print("  max-diff percentiles: p10 %.3g  p50 %.3g  p90 %.3g"
              % (diffs[len(diffs) // 10], diffs[len(diffs) // 2],
                 diffs[len(diffs) * 9 // 10]))
        near = sum(1 for d in diffs if d < 1e-2)
        print("  within 1e-2 of correct: %d (%.1f%%)" % (near, near / len(diffs) * 100))
    else:
        print("  none reached a numeric comparison")


if __name__ == "__main__":
    main()
