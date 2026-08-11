#!/usr/bin/env python3
"""Which problems a newer model lost, and how it now fails them.

A headline that goes up can hide families that went down. Stacking RL on the
distilled model gained 19 problems overall while losing 2 on activation and 1 on
loss, and giving activation a 52-task frontier quota did not bring them back --
so the question is what the newer model now does on the problems the older one
solved. "Compiles and runs but the numbers are wrong" points at forgetting;
"never produces a kernel" points at the task distribution never having taught it.

k is aligned to the smaller of the two runs, since a model measured at k=16
would otherwise get credit for attempts the other never made.

  python3 train/diagnose_regression.py --level 1 \\
      --old H:../runs/H_k16_l1 --new K:../runs/K_k4_l1 --category activation
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_partial import categorise  # noqa: E402


def load(run_dir):
    recs = json.load(open(os.path.join(run_dir, "analysis.json")))
    if isinstance(recs, dict) and "records" in recs:
        recs = recs["records"]
    by = collections.defaultdict(list)
    for r in recs:
        by[r["problem_id"]].append(r)
    for v in by.values():
        v.sort(key=lambda x: x["sample_id"])
    return by


def ok(r):
    return bool(r.get("passed")) and bool(r.get("fully_cutile"))


# Ordered by how close to working the stage is, so a problem is described by its
# best attempt rather than its worst.
#
# Do not reconstruct this from the `compiled` flag. KernelBench sets it when the
# module *imports*, and @ct.kernel does not compile eagerly -- tileiras runs at
# first launch, inside the correctness check. Treating `compiled and not correct`
# as "ran and got the numbers wrong" therefore counts every cuTile compile error
# as a numeric error. Doing exactly that put the share of unsolved problems
# blamed on numerics at 92%, when by failure_stage it is 58%, and the other 42%
# is cuTile API misuse needing an entirely different fix.
STAGE_LABEL = [
    ("wrong_numerics", "runs, wrong numbers"),
    ("cutile_frontend_error", "cuTile API misuse"),
    ("cutile_backend_compile_error", "cuTile compile error"),
    ("other_runtime_error", "runtime error"),
    ("import_error", "import error"),
    ("other", "other"),
    ("no_code_generated", "no code"),
    ("not_evaluated", "not evaluated"),
]


def why(samples):
    """How a problem failed, taken from its most-advanced attempt."""
    if any(r.get("numerically_correct") and not r.get("fully_cutile")
           for r in samples):
        return "correct but not pure"
    stages = {r.get("failure_stage") for r in samples}
    for key, label in STAGE_LABEL:
        if key in stages:
            return label
    return "unknown"


def detail(samples):
    """The specific error class, for the stages where the class is the useful bit."""
    classes = [r.get("error_class") for r in samples if r.get("error_class")]
    if not classes:
        return None
    return collections.Counter(classes).most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--old", required=True, metavar="TAG:DIR")
    ap.add_argument("--new", required=True, metavar="TAG:DIR")
    ap.add_argument("--category", default=None,
                    help="Restrict to one family; omit for all.")
    ap.add_argument("--unsolved", action="store_true",
                    help="Instead of the diff, break the newer model's unsolved "
                         "problems down by how they fail. Worth knowing before "
                         "choosing the next intervention: a set dominated by "
                         "wrong numbers needs something different from one "
                         "dominated by code that will not compile.")
    args = ap.parse_args()

    old_tag, _, old_dir = args.old.partition(":")
    new_tag, _, new_dir = args.new.partition(":")
    old, new = load(old_dir), load(new_dir)

    k = min(min(len(v) for v in old.values()), min(len(v) for v in new.values()))
    print("level %d, %s vs %s, k aligned to %d" % (args.level, old_tag, new_tag, k))

    if args.unsolved:
        by_mode = collections.Counter()
        by_cat = collections.defaultdict(collections.Counter)
        classes = collections.defaultdict(collections.Counter)
        n = 0
        for pid in sorted(new):
            name = new[pid][0].get("problem", "")
            cat = categorise(name)
            if args.category and cat != args.category:
                continue
            if any(ok(r) for r in new[pid][:k]):
                continue
            mode = why(new[pid][:k])
            by_mode[mode] += 1
            by_cat[cat][mode] += 1
            d = detail(new[pid][:k])
            if d:
                classes[mode][d] += 1
            n += 1
        print("  %s leaves %d problems unsolved at k=%d" % (new_tag, n, k))
        for mode, c in by_mode.most_common():
            top = ", ".join("%s %d" % (x, y)
                            for x, y in classes[mode].most_common(3))
            print("    %-22s %3d  %4.1f%%   %s"
                  % (mode, c, 100.0 * c / max(n, 1), top))
        print()
        print("  by family")
        for cat in sorted(by_cat, key=lambda c: -sum(by_cat[c].values())):
            tot = sum(by_cat[cat].values())
            det = ", ".join("%s %d" % (m, c)
                            for m, c in by_cat[cat].most_common())
            print("    %-12s %3d unsolved  (%s)" % (cat, tot, det))
        return

    lost, gained, both, neither = [], [], 0, 0
    for pid in sorted(set(old) & set(new)):
        name = old[pid][0].get("problem", "")
        cat = categorise(name)
        if args.category and cat != args.category:
            continue
        o = any(ok(r) for r in old[pid][:k])
        n = any(ok(r) for r in new[pid][:k])
        if o and not n:
            lost.append((pid, name, why(new[pid][:k])))
        elif n and not o:
            gained.append((pid, name, why(old[pid][:k])))
        elif o:
            both += 1
        else:
            neither += 1

    scope = args.category or "all categories"
    print("  %s: %s kept %d, %s lost %d, %s gained %d, neither solved %d"
          % (scope, new_tag, both, new_tag, len(lost), new_tag, len(gained),
             neither))

    if lost:
        print()
        print("  lost by %s -- and how it fails now:" % new_tag)
        for pid, name, reason in lost:
            print("    %-3d %-46s %s" % (pid, name[:46], reason))
        print()
        print("  failure modes: %s"
              % dict(collections.Counter(r for _, _, r in lost).most_common()))

    if gained:
        print()
        print("  gained by %s:" % new_tag)
        for pid, name, reason in gained:
            print("    %-3d %-46s (%s failed: %s)" % (pid, name[:46], old_tag,
                                                      reason))


if __name__ == "__main__":
    main()
