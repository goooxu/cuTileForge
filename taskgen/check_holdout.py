#!/usr/bin/env python3
"""Verify that a held-out task level really is disjoint from the training levels.

The generators already exclude colliding hashes as they write, so this is a
separate check on the artefact rather than a repeat of that logic: it catches the
case where the exclusion was accidentally skipped, or where a level was later
regenerated with the flag left off.

  python3 taskgen/check_holdout.py --holdout 97,98 --train 90,91,92,93,94,95,96
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_tasks import hashes_of_levels, task_hash  # noqa: E402


def level_hashes(root, level):
    d = os.path.join(root, "level%d" % level)
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            src = open(os.path.join(d, f), encoding="utf-8",
                       errors="replace").read()
            out.setdefault(task_hash(src), []).append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kernelbench/KernelBench")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--train", required=True)
    args = ap.parse_args()

    train_levels = [int(x) for x in args.train.split(",")]
    train = hashes_of_levels(args.root, train_levels)
    print("train levels %s: %d unique tasks" % (args.train, len(train)))

    holdout_levels = [int(x) for x in args.holdout.split(",")]
    seen_across = {}
    bad = 0
    for lv in holdout_levels:
        h = level_hashes(args.root, lv)
        n_files = sum(len(v) for v in h.values())
        leak = set(h) & train
        internal = {k: v for k, v in h.items() if len(v) > 1}
        cross = set(h) & set(seen_across)
        print("level%d: %d files, %d unique, %d leaked from train, "
              "%d internal dups, %d shared with earlier holdout levels"
              % (lv, n_files, len(h), len(leak), len(internal), len(cross)))
        for k in list(leak)[:5]:
            print("   leak: %s" % h[k][0])
        for k in list(internal)[:5]:
            print("   dup:  %s" % " == ".join(h[k]))
        bad += len(leak) + len(internal) + len(cross)
        seen_across.update(h)

    print("VERDICT: %s" % ("clean" if bad == 0 else "%d problems" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
