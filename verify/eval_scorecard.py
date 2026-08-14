#!/usr/bin/env python3
"""Score one or more models on the standalone eval suite.

Reads the jsonl that fast_verify.py writes. pass@1 is the per-sample rate;
pass@k is the share of problems with at least one pass in the first k samples.
Speed is the median of each problem's best compile speedup, only on the speed
track. Two models are compared on the intersection of problems both solved.

  python3 verify/eval_scorecard.py --run M:runs/eval_M
  python3 verify/eval_scorecard.py --run M:runs/eval_M --run Q:runs/eval_Q
"""
from __future__ import print_function

import argparse
import collections
import json
import os
import re
import statistics
import sys

TILE_RE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)\s*$", re.M)


def resolve_pair(prefix):
    """prefix is a run directory stem without the _lNN suffix."""
    cands = [
        (prefix + "_l60_verified.jsonl", prefix + "_l61_verified.jsonl",
         prefix + "_l61"),
        (os.path.join(prefix + "_l60", "verified.jsonl"),
         os.path.join(prefix + "_l61", "verified.jsonl"),
         prefix + "_l61"),
    ]
    for c_path, s_path, kdir in cands:
        if os.path.isfile(c_path) and os.path.isfile(s_path):
            return c_path, s_path, kdir if os.path.isdir(kdir) else None
    raise SystemExit("no verified jsonl pair under %s" % prefix)


def load_jsonl(path):
    by = collections.defaultdict(list)
    for line in open(path):
        rec = json.loads(line)
        pid = int(str(rec["key"]).split(":")[0])
        by[pid].append(rec)
    for recs in by.values():
        recs.sort(key=lambda r: int(str(r["key"]).split(":")[1]))
    return by


def pass_stats(by, k):
    n_prob = len(by)
    n_rec = sum(len(v) for v in by.values())
    n_ok = sum(1 for v in by.values() for r in v if r.get("passed"))
    solved = 0
    for recs in by.values():
        if any(r.get("passed") for r in recs[:k]):
            solved += 1
    p1 = (100.0 * n_ok / n_rec) if n_rec else 0.0
    pk = (100.0 * solved / n_prob) if n_prob else 0.0
    return n_prob, n_rec, n_ok, solved, p1, pk


def best_speedups(by):
    best = {}
    for pid, recs in by.items():
        vals = [r["speedup"] for r in recs
                if r.get("passed") and r.get("speedup")]
        if vals:
            best[pid] = max(vals)
    return best


def tile_of(src):
    found = []
    for m in TILE_RE.finditer(src):
        if any(x in m.group(1) for x in ("TILE", "BLOCK", "BM", "BN", "SIZE")):
            found.append(int(m.group(2)))
    return found[0] if found else None


def count_tiles(kernel_dir, level):
    n1024 = n256 = n_other = n_none = 0
    if not kernel_dir or not os.path.isdir(kernel_dir):
        return n1024, n256, n_other, n_none
    pat = re.compile(r"level_%d_problem_(\d+)_sample_(\d+)_kernel\.py" % level)
    for fname in os.listdir(kernel_dir):
        if not pat.match(fname):
            continue
        src = open(os.path.join(kernel_dir, fname), encoding="utf-8",
                   errors="replace").read()
        t = tile_of(src)
        if t == 1024:
            n1024 += 1
        elif t == 256:
            n256 += 1
        elif t is None:
            n_none += 1
        else:
            n_other += 1
    return n1024, n256, n_other, n_none


def score_one(tag, prefix, k):
    c_path, s_path, kdir = resolve_pair(prefix)
    c_by, s_by = load_jsonl(c_path), load_jsonl(s_path)
    c = pass_stats(c_by, k)
    s = pass_stats(s_by, k)
    speeds = best_speedups(s_by)
    med = statistics.median(speeds.values()) if speeds else None
    tiles = count_tiles(kdir, 61)
    return {
        "tag": tag, "prefix": prefix,
        "correctness": c, "speed": s,
        "n_timed": len(speeds), "median_speedup": med,
        "best_speed": speeds, "tiles": tiles,
    }


def print_row(row, k):
    c_n, _, _, c_sol, c_p1, c_pk = row["correctness"]
    s_n, _, _, s_sol, s_p1, s_pk = row["speed"]
    med = row["median_speedup"]
    t1024, t256, t_other, t_none = row["tiles"]
    med_s = ("%.3fx" % med) if med is not None else "  n/a"
    print("  %-8s  C %3d/%-3d  p@1 %5.1f%%  p@%d %5.1f%%   "
          "S %3d/%-3d  p@1 %5.1f%%  p@%d %5.1f%%  med %s  "
          "tile 1024/256/other=%d/%d/%d"
          % (row["tag"], c_sol, c_n, c_p1, k, c_pk,
             s_sol, s_n, s_p1, k, s_pk, med_s,
             t1024, t256, t_other))


def pairwise(a, b):
    common = sorted(set(a["best_speed"]) & set(b["best_speed"]))
    if not common:
        print("  no commonly solved timed speed problems")
        return
    ratios = [b["best_speed"][pid] / a["best_speed"][pid]
              for pid in common if a["best_speed"][pid] > 0]
    faster = sum(1 for r in ratios if r > 1.05)
    slower = sum(1 for r in ratios if r < 1 / 1.05)
    print("  %s vs %s on %d commonly solved speed problems: median %.3fx  "
          "faster %d  slower %d"
          % (b["tag"], a["tag"], len(common), statistics.median(ratios),
             faster, slower))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="TAG:PREFIX",
                    help="Prefix is the run stem without the _l60/_l61 suffix.")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    print("standalone eval suite  level60 correctness / level61 speed  "
          "k=%d  vs torch.compile" % args.k)
    print()
    rows = []
    for spec in args.run:
        tag, _, prefix = spec.partition(":")
        if not prefix:
            raise SystemExit("expected TAG:PREFIX, got %s" % spec)
        row = score_one(tag, prefix, args.k)
        rows.append(row)
        print_row(row, args.k)

    if len(rows) >= 2:
        print()
        pairwise(rows[0], rows[1])
        if len(rows) > 2:
            for other in rows[2:]:
                pairwise(rows[0], other)
    return 0


if __name__ == "__main__":
    sys.exit(main())
