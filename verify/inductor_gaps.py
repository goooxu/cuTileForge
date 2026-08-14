#!/usr/bin/env python3
"""Where inductor does not help, and whether we are losing there anyway.

Speed has not moved in sixteen rounds, and the last attempt failed because the
training tasks were in the wrong regime: the model already beat the reference by
6x to 11x on them, while the benchmark sits at 0.92x. Picking a better regime
needs a way to find it, and "tasks where the model is slow" turned out to select
almost entirely matmul against cuBLAS, which is unwinnable.

This asks a different question: where is the *opponent* weak? A problem on which
torch.compile is no faster than eager is one inductor could not improve, and that
is where a hand-written tile kernel has room. Crossing that with where we are
currently slower than compile gives the actual opportunity set.

  python3 verify/inductor_gaps.py --eager results/baseline_gb200_torch_fp32.json \\
      --compile results/baseline_gb200_torch_compile_fp32.json \\
      --ours ../runs/M_k4_l1:1 --ours ../runs/M_k4_l2:2
"""
import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "train"))

from compare_partial import categorise  # noqa: E402

# Below this ratio of eager to compiled time, inductor bought nothing.
INDUCTOR_FLAT = 1.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", required=True)
    ap.add_argument("--compile", dest="compiled", required=True)
    ap.add_argument("--ours", action="append", required=True,
                    metavar="RUN_DIR:LEVEL")
    ap.add_argument("--analysis-name", default="analysis_compile.json")
    args = ap.parse_args()

    eager = json.load(open(args.eager))
    comp = json.load(open(args.compiled))

    # Our best speedup per problem, against the compiled reference.
    ours = {}
    names = {}
    for spec in args.ours:
        run_dir, _, lvl = spec.rpartition(":")
        lvl = int(lvl)
        recs = json.load(open(os.path.join(run_dir, args.analysis_name)))
        if isinstance(recs, dict) and "records" in recs:
            recs = recs["records"]
        for r in recs:
            if r.get("passed") and r.get("fully_cutile") and r.get("speedup"):
                key = (lvl, r["problem"])
                ours[key] = max(ours.get(key, 0.0), r["speedup"])
                names[key] = r["problem"]

    rows = []
    for lvl in (1, 2):
        ekey, ckey = "level%d" % lvl, "level%d" % lvl
        for name, ev in eager.get(ekey, {}).items():
            cv = comp.get(ckey, {}).get(name)
            if not (isinstance(ev, dict) and isinstance(cv, dict)
                    and "mean" in ev and "mean" in cv and cv["mean"] > 0):
                continue
            inductor_gain = ev["mean"] / cv["mean"]
            rows.append((lvl, name, inductor_gain, ours.get((lvl, name))))

    flat = [r for r in rows if r[2] < INDUCTOR_FLAT]
    helped = [r for r in rows if r[2] >= INDUCTOR_FLAT]
    print("%d problems with both baselines timed" % len(rows))
    print("  inductor helped (>=%.2fx over eager)  %d" % (INDUCTOR_FLAT, len(helped)))
    print("  inductor flat                          %d" % len(flat))

    def summarise(label, rs):
        solved = [r for r in rs if r[3]]
        if not solved:
            print("  %-22s no solved problems" % label)
            return
        sp = sorted(r[3] for r in solved)
        beat = sum(1 for s in sp if s > 1.0)
        print("  %-22s solved %3d/%-3d  median %.3fx  we beat compile on %d"
              % (label, len(solved), len(rs), statistics.median(sp), beat))

    print()
    print("  our speed on each group, against the compiled reference")
    summarise("inductor flat", flat)
    summarise("inductor helped", helped)

    # The opportunity set: inductor could not help, and we are still slower.
    opp = [r for r in flat if r[3] and r[3] < 1.0]
    print()
    print("  opportunity set (inductor flat AND we are slower): %d problems" % len(opp))
    cats = collections.Counter(categorise(r[1]) for r in opp)
    print("    by family: %s" % dict(cats.most_common()))
    opp.sort(key=lambda r: r[3])
    for lvl, name, gain, sp in opp[:12]:
        print("    L%d %-52s ours %.3fx" % (lvl, name[:52], sp))

    # And where inductor is flat and we already win -- the shape to generate more of.
    win = [r for r in flat if r[3] and r[3] > 1.0]
    print()
    print("  where inductor is flat and we already win: %d problems" % len(win))
    print("    by family: %s"
          % dict(collections.Counter(categorise(r[1]) for r in win).most_common()))


if __name__ == "__main__":
    main()
