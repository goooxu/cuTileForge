#!/usr/bin/env python3
"""Did changing the prompt tile from 256 to 1024 move the activation anchors?

Reads a timed harvest of Level-1 activations and compares each problem's best
speedup to model M's published analysis. Also counts which TILE_SIZE the new
kernels actually wrote -- the prompt change is a miss if they still emit 256.

  python3 verify/compare_tile1024.py \
      --verified runs/tile1024_l1act_verified.jsonl \
      --kernel-dir runs/tile1024_l1act \
      --baseline results/level1_per_sample_compile.json
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

ANCHORS = ("27_SELU_", "31_ELU", "32_HardTanh")
SANITY = ("88_MinGPTNewGelu",)
# Softmax-family is a reduction; keep it out of the pointwise headline.
SKIP = ("23_Softmax", "24_LogSoftmax")
PID_NAMES = {
    19: "19_ReLU.py", 20: "20_LeakyReLU.py", 21: "21_Sigmoid.py",
    22: "22_Tanh.py", 23: "23_Softmax.py", 24: "24_LogSoftmax.py",
    25: "25_Swish.py", 26: "26_GELU_.py", 27: "27_SELU_.py",
    28: "28_HardSigmoid.py", 29: "29_Softplus.py", 30: "30_Softsign.py",
    31: "31_ELU.py", 32: "32_HardTanh.py", 88: "88_MinGPTNewGelu.py",
}
TILE_RE = re.compile(
    r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)\s*$", re.M)
KERNEL_RE = re.compile(r"level_1_problem_(\d+)_sample_(\d+)_kernel\.py")


def load_m(path):
    recs = json.load(open(path))
    if isinstance(recs, dict) and "records" in recs:
        recs = recs["records"]
    best, names, by_id = {}, {}, {}
    for r in recs:
        if not (r.get("passed") and r.get("fully_cutile") and r.get("speedup")):
            continue
        name = r.get("problem") or str(r.get("problem_id"))
        pid = int(r["problem_id"])
        best[name] = max(best.get(name, 0.0), r["speedup"])
        by_id[pid] = max(by_id.get(pid, 0.0), r["speedup"])
        names[pid] = name
    return best, names, by_id


def tile_of(src):
    found = []
    for m in TILE_RE.finditer(src):
        if any(x in m.group(1) for x in ("TILE", "BLOCK", "BM", "BN", "SIZE")):
            found.append(int(m.group(2)))
    return found[0] if found else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()

    m_best, id_to_name, m_by_id = load_m(args.baseline)
    # Fall back to reading names from the harvest records if M used a
    # different analysis shape.
    by = collections.defaultdict(list)
    for line in open(args.verified):
        rec = json.loads(line)
        pid = int(rec["key"].split(":")[0])
        sid = int(rec["key"].split(":")[1])
        rec["_pid"] = pid
        rec["_sid"] = sid
        by[pid].append(rec)

    tiles = {}
    tile_counts = collections.Counter()
    for fname in os.listdir(args.kernel_dir):
        m = KERNEL_RE.match(fname)
        if not m:
            continue
        src = open(os.path.join(args.kernel_dir, fname),
                   encoding="utf-8", errors="replace").read()
        t = tile_of(src)
        tiles[(int(m.group(1)), int(m.group(2)))] = t
        tile_counts[t if t is not None else "(none)"] += 1

    print("kernels %d   tile literals: %s"
          % (sum(tile_counts.values()), tile_counts.most_common(8)))
    n256 = tile_counts.get(256, 0)
    n1024 = tile_counts.get(1024, 0)
    print("  wrote 256: %d    wrote 1024: %d"
          % (n256, n1024))

    def name_of(pid):
        if pid in PID_NAMES:
            return PID_NAMES[pid]
        if pid in id_to_name:
            return id_to_name[pid]
        recs = by[pid]
        for r in recs:
            if r.get("problem"):
                return r["problem"]
        return "problem_%d" % pid

    rows = []
    for pid, recs in sorted(by.items()):
        name = name_of(pid)
        if any(s in name for s in SKIP):
            kind = "softmax"
        elif any(s in name for s in ANCHORS):
            kind = "anchor"
        elif any(s in name for s in SANITY):
            kind = "sanity"
        else:
            kind = "pointwise"
        passed = [r for r in recs if r.get("passed") and r.get("speedup")]
        best = max((r["speedup"] for r in passed), default=None)
        used = [tiles.get((pid, r["_sid"])) for r in recs]
        rows.append((kind, pid, name, best, used, passed))

    print()
    print("%-8s %-40s  new     M      ratio   tiles"
          % ("kind", "problem"))
    ratios = []
    anchor_new = {}
    for kind, pid, name, best, used, passed in rows:
        old = m_by_id.get(pid)
        if old is None:
            for key, val in m_best.items():
                if name in key or key in name:
                    old = val
                    break
        ratio = (best / old) if (best and old) else None
        if kind == "pointwise" and ratio:
            ratios.append(ratio)
        if kind == "anchor" and best is not None:
            anchor_new[name] = best
        tile_s = ",".join(str(t) if t else "?" for t in used)
        print("%-8s %-40s  %s  %s  %s   %s"
              % (kind, name[:40],
                 ("%.3fx" % best) if best else "  --- ",
                 ("%.3fx" % old) if old else "  --- ",
                 ("%.3fx" % ratio) if ratio else "  --- ",
                 tile_s))

    print()
    if ratios:
        print("pointwise pairwise new/M: n=%d  median %.3fx"
              % (len(ratios), statistics.median(ratios)))
    print("anchors:")
    for prefix in ANCHORS:
        hits = [(n, s) for n, s in anchor_new.items() if prefix.lower() in n.lower()]
        if not hits:
            print("  %-20s  not solved" % prefix)
            continue
        for n, s in hits:
            print("  %-20s  %.3fx   %s"
                  % (n, s, "PASS >=0.80x" if s >= 0.80 else "still below 0.80x"))


if __name__ == "__main__":
    main()
