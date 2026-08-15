#!/usr/bin/env python3
"""Score one or more models on the standalone eval suite.

Reads the jsonl that fast_verify.py writes. pass@1 is the per-sample rate;
pass@k is the share of problems with at least one pass in the first k samples.

Latency (770 graphs) and throughput twins are never folded into one median.
Each is split common / awkward. Two models are compared on the intersection
of problems both solved.

  python3 verify/eval_scorecard.py --run M:runs/M
  python3 verify/eval_scorecard.py --run M:runs/M --run Q:runs/Q
"""
from __future__ import print_function

import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import INCONCLUSIVE_STAGES  # noqa: E402

TILE_RE = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)\s*$", re.M)
THRESHOLD = 1.05


def forge_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_manifest(path):
    man = json.load(open(path))
    by = {}
    for p in man["problems"]:
        by[int(p["problem_id"])] = p
    return man, by


def resolve_run(prefix):
    """prefix is a run directory stem without the _lNN suffix."""
    cands = [
        (prefix + "_l60_verified.jsonl", prefix + "_l60"),
        (os.path.join(prefix + "_l60", "verified.jsonl"), prefix + "_l60"),
        (prefix + "_verified.jsonl", prefix),
    ]
    for path, kdir in cands:
        if os.path.isfile(path):
            return path, kdir if os.path.isdir(kdir) else None
    raise SystemExit("no verified jsonl under %s" % prefix)


def load_jsonl(path):
    by = collections.defaultdict(list)
    for line in open(path):
        rec = json.loads(line)
        pid = int(str(rec["key"]).split(":")[0])
        by[pid].append(rec)
    for recs in by.values():
        recs.sort(key=lambda r: int(str(r["key"]).split(":")[1]))
    return by


def select(by, man_by, role=None, kind=None):
    out = {}
    for pid, recs in by.items():
        meta = man_by.get(pid)
        if meta is None:
            continue
        if role is not None and meta.get("role") != role:
            continue
        if kind is not None and meta.get("shape_kind") != kind:
            continue
        out[pid] = recs
    return out


def pass_stats(by, k):
    n_prob = len(by)
    n_rec = sum(1 for v in by.values() for r in v
                if r.get("stage") not in INCONCLUSIVE_STAGES)
    n_ok = sum(1 for v in by.values() for r in v if r.get("passed"))
    solved = 0
    for recs in by.values():
        if any(r.get("passed") for r in recs[:k]):
            solved += 1
    p1 = (100.0 * n_ok / n_rec) if n_rec else 0.0
    pk = (100.0 * solved / n_prob) if n_prob else 0.0
    return n_prob, n_rec, n_ok, solved, p1, pk


def best_timed(by):
    best = {}
    for pid, recs in by.items():
        cand = [r for r in recs
                if r.get("passed") and r.get("speedup") and r.get("kernel_ms")]
        if not cand:
            continue
        rec = max(cand, key=lambda r: r["speedup"])
        best[pid] = {"speedup": rec["speedup"], "kernel_ms": rec["kernel_ms"]}
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


def score_one(tag, prefix, k, man_by):
    path, kdir = resolve_run(prefix)
    by = load_jsonl(path)
    lat = select(by, man_by, role="latency")
    thr = select(by, man_by, role="throughput")
    return {
        "tag": tag, "prefix": prefix, "by": by,
        "latency": pass_stats(lat, k),
        "throughput": pass_stats(thr, k),
        "best_lat": best_timed(lat),
        "best_thr": best_timed(thr),
        "best_lat_common": best_timed(select(by, man_by, "latency", "common")),
        "best_lat_awkward": best_timed(select(by, man_by, "latency", "awkward")),
        "best_thr_common": best_timed(select(by, man_by, "throughput", "common")),
        "best_thr_awkward": best_timed(select(by, man_by, "throughput", "awkward")),
        "tiles": count_tiles(kdir, 60),
        "solved_lat": set(pid for pid, recs in lat.items()
                          if any(r.get("passed") for r in recs[:k])),
    }


def print_row(row, k):
    l_n, _, _, l_sol, l_p1, l_pk = row["latency"]
    t_n, _, _, t_sol, t_p1, t_pk = row["throughput"]
    t1024, t256, t_other, t_none = row["tiles"]
    print("  %-8s  L %3d/%-3d  p@1 %5.1f%%  p@%d %5.1f%%   "
          "T %3d/%-3d  p@1 %5.1f%%  p@%d %5.1f%%  "
          "tile 1024/256/other=%d/%d/%d"
          % (row["tag"], l_sol, l_n, l_p1, k, l_pk,
             t_sol, t_n, t_p1, k, t_pk,
             t1024, t256, t_other))


def pairwise(a_best, b_best, a_tag, b_tag, label):
    common = sorted(set(a_best) & set(b_best))
    if not common:
        print("  %s vs %s  %-8s  no commonly solved timed problems"
              % (b_tag, a_tag, label))
        return
    su = [b_best[pid]["speedup"] / a_best[pid]["speedup"]
          for pid in common if a_best[pid]["speedup"] > 0]
    ms = [a_best[pid]["kernel_ms"] / b_best[pid]["kernel_ms"]
          for pid in common if b_best[pid]["kernel_ms"] > 0]
    faster = sum(1 for r in su if r > THRESHOLD)
    slower = sum(1 for r in su if r < 1 / THRESHOLD)
    med_su = statistics.median(su) if su else float("nan")
    med_ms = statistics.median(ms) if ms else float("nan")
    print("  %s vs %s  %-8s  n=%-3d  med su %.3fx  med ms %.3fx  "
          "faster %d  slower %d"
          % (b_tag, a_tag, label, len(common), med_su, med_ms,
             faster, slower))


def print_pairwise_block(rows, key_all, key_c, key_a, title):
    print()
    print(title)
    if len(rows) < 2:
        return
    pairs = [(rows[0], rows[1])]
    if len(rows) > 2:
        pairs.append((rows[0], rows[2]))
        pairs.append((rows[1], rows[2]))
    for a, b in pairs:
        pairwise(a[key_all], b[key_all], a["tag"], b["tag"], "all")
        pairwise(a[key_c], b[key_c], a["tag"], b["tag"], "common")
        pairwise(a[key_a], b[key_a], a["tag"], b["tag"], "awkward")


def family_table(rows, man_by, k):
    cats = collections.OrderedDict()
    srcs = collections.OrderedDict()
    for pid, meta in man_by.items():
        if meta.get("role") != "latency":
            continue
        cats.setdefault(meta.get("category") or "?", []).append(pid)
        srcs.setdefault(str(meta.get("source_level")), []).append(pid)
    print()
    print("correctness by family (solved @k=%d, latency graphs only)" % k)
    print("  %-12s %4s  %s" % ("family", "n", "  ".join(
        "%-8s" % r["tag"] for r in rows)))
    for cat, pids in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        cells = []
        for r in rows:
            n = sum(1 for pid in pids if pid in r["solved_lat"])
            cells.append("%3d/%-3d" % (n, len(pids)))
        print("  %-12s %4d  %s" % (cat, len(pids), "  ".join(
            "%-8s" % c for c in cells)))
    print()
    print("correctness by source (solved @k=%d)" % k)
    for src, pids in sorted(srcs.items()):
        cells = []
        for r in rows:
            n = sum(1 for pid in pids if pid in r["solved_lat"])
            cells.append("%3d/%-3d" % (n, len(pids)))
        print("  level %-4s n=%-3d  %s" % (src, len(pids), "  ".join(
            "%s %s" % (r["tag"], c) for r, c in zip(rows, cells))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="TAG:PREFIX",
                    help="Prefix is the run stem without the _l60 suffix.")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--manifest", default=None,
                    help="Default: tasks/eval/manifest.json next to this repo.")
    args = ap.parse_args()

    man_path = args.manifest or os.path.join(
        forge_root(), "tasks", "eval", "manifest.json")
    if not os.path.isfile(man_path):
        raise SystemExit("manifest not found: %s" % man_path)
    man, man_by = load_manifest(man_path)
    n_lat = man.get("n_latency", sum(
        1 for p in man["problems"] if p.get("role") == "latency"))
    n_thr = man.get("n_throughput", sum(
        1 for p in man["problems"] if p.get("role") == "throughput"))

    print("standalone eval suite  level 60  k=%d  vs torch.compile" % args.k)
    print("%d latency + %d throughput  (do not fold these medians together)"
          % (n_lat, n_thr))
    print()
    rows = []
    for spec in args.run:
        tag, _, prefix = spec.partition(":")
        if not prefix:
            raise SystemExit("expected TAG:PREFIX, got %s" % spec)
        row = score_one(tag, prefix, args.k, man_by)
        rows.append(row)
        print_row(row, args.k)

    print_pairwise_block(
        rows, "best_lat", "best_lat_common", "best_lat_awkward",
        "latency pairwise  (su = compile speedup ratio; "
        "ms = kernel_ms ratio, >1 means second tag is faster)")
    print_pairwise_block(
        rows, "best_thr", "best_thr_common", "best_thr_awkward",
        "throughput pairwise  (same ratios; awkward collapse is a signal)")
    family_table(rows, man_by, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
