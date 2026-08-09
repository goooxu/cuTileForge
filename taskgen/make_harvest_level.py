#!/usr/bin/env python3
"""Assemble one KernelBench level by sampling across several existing levels.

Generation and evaluation both work a level at a time, so drawing a mixed
subset of the training tasks means materialising it as its own level. Problem
ids are renumbered from 1 because the dataset loader keys on the leading integer
in the filename, and a manifest records where each task came from so a result on
this level can be traced back to the ladder it was drawn from.

  python3 taskgen/make_harvest_level.py --from 90,91,92,93,94,95,96 \\
      --count 1000 --level 89 --seed 100089
"""
import argparse
import json
import os
import random
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="kernelbench/KernelBench")
    ap.add_argument("--from", dest="src", required=True,
                    help="Comma-separated source levels.")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--level", type=int, required=True, help="Destination level.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    pool = []
    for lvl in [int(x) for x in args.src.split(",")]:
        d = os.path.join(args.root, "level%d" % lvl)
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                pool.append((lvl, f))
    print("pool: %d tasks across levels %s" % (len(pool), args.src))

    rng = random.Random(args.seed)
    picked = rng.sample(pool, min(args.count, len(pool)))
    # Sort so the level reads in a stable order; the sample itself is the random
    # part and re-sorting does not bias it.
    picked.sort()

    out_dir = os.path.join(args.root, "level%d" % args.level)
    if args.clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for i, (lvl, fname) in enumerate(picked, start=1):
        # Strip the source id; keep the descriptive part.
        stem = fname[:-3]
        _, _, rest = stem.partition("_")
        new_name = "%d_%s.py" % (i, rest or stem)
        shutil.copyfile(os.path.join(args.root, "level%d" % lvl, fname),
                        os.path.join(out_dir, new_name))
        manifest.append({"problem_id": i, "file": new_name,
                         "source_level": lvl, "source_file": fname})

    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    import collections
    by_src = collections.Counter(m["source_level"] for m in manifest)
    print("wrote %d tasks to %s" % (len(manifest), out_dir))
    print("  by source level: %s" % dict(sorted(by_src.items())))


if __name__ == "__main__":
    main()
