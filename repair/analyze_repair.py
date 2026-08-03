"""Conversion-rate analysis for the multi-turn repair loop.

The decisive question is whether feeding the compiler's diagnostic back to the
model breaks convolution's cold start: 263 conv samples in phase two produced
zero correct kernels, so rejection sampling had no seed to bootstrap from.

Beyond the headline rate this separates two things that look alike in the
aggregate: candidates that were genuinely repaired, and candidates that merely
traded one failure for another. A model that reacts to "grid must be at most 3
dimensions" by cutting the grid to 3 and getting the indexing wrong has not
learnt anything useful, and its trajectory is not worth training on.

Usage:
    python3 repair/analyze_repair.py --run runs/repair_l93 --level 93
"""

import argparse
import collections
import json
import os
import re


def classify(stage: str, error: str) -> str:
    """Bucket a failure by cuTile error signature.

    The buckets match those used in the phase-two analysis so the numbers stay
    comparable.
    """
    if stage in ("timeout", "oom", "no_code", "purity"):
        return stage
    e = error or ""
    pats = [
        (r"Grid dimensions must be at most 3", "grid_rank_exceeded"),
        (r"Expected shape length to be \d+, got \d+", "rank_mismatch"),
        (r"No such attribute '(\w+)' for object of type Array", "array_used_as_tensor"),
        (r"[Ii]ncompatible shapes|shape .* != expected", "shape_mismatch"),
        (r"output mismatch", "numeric_mismatch"),
        (r"non-finite", "non_finite"),
        (r"No such attribute", "bad_attribute"),
        (r"Invalid argument", "invalid_argument"),
        (r"NameError|not defined", "name_error"),
        (r"TypeError", "type_error"),
        (r"SyntaxError|IndentationError", "syntax_error"),
    ]
    for pat, name in pats:
        if re.search(pat, e):
            return name
    return "other"


def task_category(problem: str) -> str:
    p = problem.lower()
    for key, cat in (("conv", "conv"), ("pool", "pool"), ("norm", "norm"),
                     ("matmul", "matmul"), ("softmax", "norm")):
        if key in p:
            return cat
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--level", type=int, default=93)
    args = ap.parse_args()

    traj_path = os.path.join(args.run, "trajectories.jsonl")
    trajs = [json.loads(l) for l in open(traj_path)]

    # Category comes from the generated problem's own docstring where possible.
    cat_of = {}
    prob_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "kernelbench", "KernelBench",
                            "level%d" % args.level)
    if os.path.isdir(prob_dir):
        for f in os.listdir(prob_dir):
            m = re.match(r"(\d+)_", f)
            if not m:
                continue
            d = re.search(r'"""(\w+) \(tier (\d+), (\w+)\)', open(
                os.path.join(prob_dir, f)).read())
            if d:
                cat_of[int(m.group(1))] = (d.group(3), int(d.group(2)))

    n = len(trajs)
    max_round = max((len(t["history"]) for t in trajs), default=1)

    print("=" * 66)
    print("REPAIR LOOP: %d candidates, up to %d rounds" % (n, max_round - 1))
    print("=" * 66)

    # ---- cumulative pass rate by round -----------------------------------
    print("\n### cumulative pass rate by round\n")
    print("  round      newly passed   cumulative   rate")
    cum = 0
    by_round = collections.Counter(t["passed_round"] for t in trajs
                                   if t["passed_round"] is not None)
    for r in range(max_round):
        cum += by_round.get(r, 0)
        label = "initial" if r == 0 else "+repair %d" % r
        print("  %-10s %8d     %8d   %5.1f%%"
              % (label, by_round.get(r, 0), cum, cum / n * 100))

    # ---- by category ------------------------------------------------------
    print("\n### by category: initial vs after repair\n")
    print("  %-10s %6s %10s %10s %10s" % ("category", "cands", "initial",
                                          "final", "delta"))
    cat_tot = collections.Counter()
    cat_init = collections.Counter()
    cat_final = collections.Counter()
    for t in trajs:
        cat = cat_of.get(t["problem_id"], (task_category(t["problem"]), -1))[0]
        cat_tot[cat] += 1
        if t["passed_round"] == 0:
            cat_init[cat] += 1
        if t["passed_round"] is not None:
            cat_final[cat] += 1
    for cat, tot in cat_tot.most_common():
        i, fin = cat_init[cat], cat_final[cat]
        print("  %-10s %6d %9.1f%% %9.1f%% %+9.1fpp"
              % (cat, tot, i / tot * 100, fin / tot * 100,
                 (fin - i) / tot * 100))

    # ---- conv detail: the decisive metric --------------------------------
    conv = [t for t in trajs
            if cat_of.get(t["problem_id"], (task_category(t["problem"]), -1))[0] == "conv"]
    if conv:
        conv_pass = [t for t in conv if t["passed_round"] is not None]
        print("\n### convolution (phase-two baseline: 0/263 = 0.0%%)\n")
        print("  candidates            %d" % len(conv))
        print("  passed at some round  %d (%.1f%%)"
              % (len(conv_pass), len(conv_pass) / len(conv) * 100))
        print("  distinct problems     %d of %d"
              % (len({t["problem_id"] for t in conv_pass}),
                 len({t["problem_id"] for t in conv})))
        if conv_pass:
            rr = collections.Counter(t["passed_round"] for t in conv_pass)
            print("  by round              %s"
                  % ", ".join("r%d:%d" % (k, rr[k]) for k in sorted(rr)))
            tiers = collections.Counter(
                cat_of.get(t["problem_id"], ("conv", -1))[1] for t in conv_pass)
            print("  by tier               %s"
                  % ", ".join("t%d:%d" % (k, tiers[k]) for k in sorted(tiers)))
            ops = collections.Counter(t["problem"] for t in conv_pass)
            print("  operators             %s"
                  % ", ".join("%s:%d" % (k, v) for k, v in ops.most_common(6)))

    # ---- repair success rate by error class ------------------------------
    print("\n### repair success by error class (of the failure being repaired)\n")
    print("  %-22s %8s %8s %8s" % ("error class", "seen", "fixed", "rate"))
    seen = collections.Counter()
    fixed = collections.Counter()
    for t in trajs:
        h = t["history"]
        for i, step in enumerate(h[:-1]):
            if step["passed"]:
                continue
            cls = classify(step["stage"], step["error"])
            seen[cls] += 1
            if h[i + 1]["passed"]:
                fixed[cls] += 1
    for cls, s in seen.most_common(14):
        print("  %-22s %8d %8d %7.1f%%" % (cls, s, fixed[cls], fixed[cls] / s * 100))

    # ---- genuinely repaired vs just a different failure ------------------
    print("\n### what a repair attempt actually did\n")
    outcome = collections.Counter()
    for t in trajs:
        h = t["history"]
        for i, step in enumerate(h[:-1]):
            if step["passed"]:
                continue
            nxt = h[i + 1]
            if nxt["passed"]:
                outcome["fixed"] += 1
                continue
            a = classify(step["stage"], step["error"])
            b = classify(nxt["stage"], nxt["error"])
            if a == b:
                outcome["same error again"] += 1
            elif a != "numeric_mismatch" and b == "numeric_mismatch":
                # Compiles now but computes the wrong thing: progress of a sort,
                # but it is not a working kernel.
                outcome["now compiles, wrong numbers"] += 1
            else:
                outcome["different error"] += 1
    tot_att = sum(outcome.values())
    for k, v in outcome.most_common():
        print("  %-28s %6d  %5.1f%%" % (k, v, v / max(tot_att, 1) * 100))

    # ---- harvested pool ---------------------------------------------------
    kernels = [f for f in os.listdir(args.run) if f.endswith("_kernel.py")]
    print("\n### harvested positive pool\n")
    print("  kernels written       %d" % len(kernels))
    print("  distinct problems     %d"
          % len({t["problem_id"] for t in trajs if t["passed_round"] is not None}))
    from_repair = sum(1 for t in trajs
                      if t["passed_round"] is not None and t["passed_round"] > 0)
    print("  of which from repair  %d (%.1f%% of the pool)"
          % (from_repair, from_repair / max(len(kernels), 1) * 100))


if __name__ == "__main__":
    main()
