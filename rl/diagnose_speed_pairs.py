#!/usr/bin/env python3
"""CPU diagnosis of within-problem kernel_ms pairs from the GL-E harvest.

Does not train. Writes a filtered pair list and prints whether the fast/slow
gap looks like a writing choice (tile, mma, length) or like timing junk.

  python3 rl/diagnose_speed_pairs.py
"""
from __future__ import print_function

import argparse
import collections
import json
import os
import re
import statistics
import sys

TILE_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*$", re.M)
SHAPE_RE = re.compile(r"shape\s*=\s*\((\d+)\s*,\s*(\d+)\)")
CAT_RE = re.compile(r'"""(\w+) \(tier (\d+), (\w+)\)')
API_MARKERS = (
    "ct.mma", "ct.matmul", "ct.gemm", "ct.conv",
    "PaddingMode", "ct.load", "ct.store",
)
TILE_NAME = re.compile(
    r"(TILE|BLOCK|BM|BN|BK|TM|TN|TK|TH|TW|SIZE)")


def forge_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def category_of(src):
    m = CAT_RE.search(src)
    return m.group(3) if m else "?"


def load_cats(root, level):
    lvdir = os.path.join(root, "level%d" % level)
    cats = {}
    if not os.path.isdir(lvdir):
        return cats
    for fn in os.listdir(lvdir):
        if not fn.endswith(".py"):
            continue
        pid = int(fn.split("_", 1)[0])
        cats[pid] = category_of(
            open(os.path.join(lvdir, fn), encoding="utf-8", errors="replace").read())
    return cats


def normalise(code):
    out = []
    for line in code.splitlines():
        line = re.sub(r"#.*$", "", line).rstrip()
        if line.strip():
            out.append(re.sub(r"\s+", " ", line.strip()))
    return "\n".join(out)


def tiles_of(src):
    found = []
    for m in TILE_RE.finditer(src):
        if TILE_NAME.search(m.group(1)):
            found.append((m.group(1), int(m.group(2))))
    if not found:
        for m in SHAPE_RE.finditer(src):
            found.append(("shape", int(m.group(1))))
            found.append(("shape", int(m.group(2))))
    return found


def features(src):
    t = tiles_of(src)
    vals = [v for _, v in t]
    return {
        "n_chars": len(src),
        "n_lines": src.count("\n") + 1,
        "tiles": t,
        "tile_vals": tuple(sorted(set(vals))),
        "has_1024": 1024 in vals,
        "has_256": 256 in vals,
        "mma": "ct.mma" in src,
        "matmul": "ct.matmul" in src or "ct.gemm" in src,
        "conv": "ct.conv" in src,
        "padding": "PaddingMode" in src,
        "n_load": src.count("ct.load"),
        "n_store": src.count("ct.store"),
        "n_kernel": src.count("@ct.kernel") + src.count("@cutile.kernel"),
        "api": [a for a in API_MARKERS if a in src],
    }


def load_harvest(ws, levels):
    by = collections.defaultdict(list)
    for lv in levels:
        path = os.path.join(ws, "runs", "harvest_gle%d_verified.jsonl" % lv)
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("passed") or rec.get("kernel_ms") is None:
                continue
            if rec["kernel_ms"] <= 0:
                continue
            pid, sid = rec["key"].split(":")
            by[(lv, int(pid))].append({
                "sid": int(sid),
                "kernel_ms": rec["kernel_ms"],
                "speedup": rec.get("speedup"),
            })
    return by


def read_kernel(ws, level, pid, sid):
    path = os.path.join(
        ws, "runs", "harvest_gle%d" % level,
        "level_%d_problem_%d_sample_%d_kernel.py" % (level, pid, sid))
    if not os.path.isfile(path):
        return None, path
    return open(path, encoding="utf-8", errors="replace").read(), path


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default=os.path.dirname(forge_root()))
    ap.add_argument("--min-spread", type=float, default=1.5)
    ap.add_argument("--max-spread", type=float, default=8.0)
    ap.add_argument("--min-ms", type=float, default=0.05)
    ap.add_argument("--max-ms", type=float, default=50.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ws = args.ws
    forge = forge_root()
    kb = os.path.join(forge, "kernelbench", "KernelBench")
    levels = (86, 87, 92, 93)
    cats = {}
    for lv in levels:
        cats.update(((lv, pid), c) for pid, c in load_cats(kb, lv).items())

    harvest = load_harvest(ws, levels)
    raw_pairs = []
    for (lv, pid), recs in harvest.items():
        if len(recs) < 2:
            continue
        recs = sorted(recs, key=lambda r: (r["kernel_ms"], r["sid"]))
        fast, slow = recs[0], recs[-1]
        spread = slow["kernel_ms"] / fast["kernel_ms"]
        raw_pairs.append({
            "level": lv, "pid": pid, "cat": cats.get((lv, pid), "?"),
            "fast": fast, "slow": slow, "spread": spread,
            "n_timed": len(recs),
            "best_su": max(r["speedup"] for r in recs if r.get("speedup")),
        })

    def in_band(p):
        if p["spread"] < args.min_spread or p["spread"] > args.max_spread:
            return False
        for side in ("fast", "slow"):
            ms = p[side]["kernel_ms"]
            if ms < args.min_ms or ms > args.max_ms:
                return False
        return True

    band = [p for p in raw_pairs if in_band(p)]
    print("timed problems with >=2 passes: %d" % len(raw_pairs))
    print("band spread [%.1f, %.1f] and kernel_ms [%.2f, %.1f]: %d"
          % (args.min_spread, args.max_spread, args.min_ms, args.max_ms, len(band)))
    if raw_pairs:
        spreads = sorted(p["spread"] for p in raw_pairs)
        print("  all-pair spread  median %.2fx  p90 %.1fx  max %.0fx"
              % (spreads[len(spreads) // 2],
                 spreads[int(0.9 * (len(spreads) - 1))], spreads[-1]))

    fam = collections.Counter(p["cat"] for p in band)
    gemm = fam.get("conv", 0) + fam.get("matmul", 0)
    print("  families %s" % dict(fam))
    print("  GEMM %d (%.0f%%)" % (gemm, pct(gemm, len(band))))

    kept = []
    n_dup = n_missing = 0
    feat_diff = collections.Counter()
    fast_wins = collections.Counter()
    for p in band:
        fc, fp = read_kernel(ws, p["level"], p["pid"], p["fast"]["sid"])
        sc, spath = read_kernel(ws, p["level"], p["pid"], p["slow"]["sid"])
        if fc is None or sc is None:
            n_missing += 1
            continue
        if normalise(fc) == normalise(sc):
            n_dup += 1
            continue
        ff, sf = features(fc), features(sc)
        rec = {
            "level": p["level"], "problem_id": p["pid"], "category": p["cat"],
            "spread": round(p["spread"], 4),
            "n_timed": p["n_timed"],
            "best_speedup": p["best_su"],
            "chosen": {"sample_id": p["fast"]["sid"],
                       "kernel_ms": p["fast"]["kernel_ms"],
                       "path": fp, "feat": ff},
            "rejected": {"sample_id": p["slow"]["sid"],
                         "kernel_ms": p["slow"]["kernel_ms"],
                         "path": spath, "feat": sf},
        }
        kept.append(rec)
        if ff["tile_vals"] != sf["tile_vals"]:
            feat_diff["tile"] += 1
        if ff["mma"] != sf["mma"]:
            feat_diff["mma"] += 1
        if ff["matmul"] != sf["matmul"]:
            feat_diff["matmul"] += 1
        if abs(ff["n_chars"] - sf["n_chars"]) >= 200:
            feat_diff["len>=200"] += 1
        if ff["api"] != sf["api"]:
            feat_diff["api"] += 1
        if ff["has_1024"] and not sf["has_1024"]:
            fast_wins["only_fast_has_1024"] += 1
        if sf["has_1024"] and not ff["has_1024"]:
            fast_wins["only_slow_has_1024"] += 1
        if ff["mma"] and not sf["mma"]:
            fast_wins["only_fast_mma"] += 1
        if sf["mma"] and not ff["mma"]:
            fast_wins["only_slow_mma"] += 1
        if ff["n_chars"] + 200 < sf["n_chars"]:
            fast_wins["fast_shorter"] += 1
        elif sf["n_chars"] + 200 < ff["n_chars"]:
            fast_wins["fast_longer"] += 1
        # Product of explicit tile constants: smaller tile often wins on
        # latency-sized GEMM (TM=64 vs TM=256 on the same mma loop).
        fp = 1
        for v in ff["tile_vals"] or (1,):
            fp *= v
        sp = 1
        for v in sf["tile_vals"] or (1,):
            sp *= v
        if ff["tile_vals"] and sf["tile_vals"] and fp != sp:
            if fp < sp:
                fast_wins["fast_smaller_tileprod"] += 1
            else:
                fast_wins["fast_larger_tileprod"] += 1

    print("  missing kernel file: %d" % n_missing)
    print("  identical after norm: %d" % n_dup)
    print("  distinct-code pairs: %d" % len(kept))
    n = len(kept) or 1
    print("  feature disagreements:")
    for k in ("tile", "mma", "matmul", "api", "len>=200"):
        print("    %-12s %4d  %4.1f%%" % (k, feat_diff[k], pct(feat_diff[k], n)))
    print("  which side has the feature:")
    for k in ("only_fast_has_1024", "only_slow_has_1024",
              "only_fast_mma", "only_slow_mma",
              "fast_shorter", "fast_longer",
              "fast_smaller_tileprod", "fast_larger_tileprod"):
        print("    %-22s %4d" % (k, fast_wins[k]))

    # GEMM-only feature split: that is what table A latency can use.
    gemm_kept = [r for r in kept if r["category"] in ("conv", "matmul")]
    print("  GEMM distinct pairs: %d" % len(gemm_kept))
    if gemm_kept:
        g_tile = sum(1 for r in gemm_kept
                     if r["chosen"]["feat"]["tile_vals"]
                     != r["rejected"]["feat"]["tile_vals"])
        g_mma = sum(1 for r in gemm_kept
                    if r["chosen"]["feat"]["mma"] != r["rejected"]["feat"]["mma"])
        print("    tile disagree %d (%.0f%%)  mma disagree %d (%.0f%%)"
              % (g_tile, pct(g_tile, len(gemm_kept)),
                 g_mma, pct(g_mma, len(gemm_kept))))
        tile_pairs = collections.Counter()
        for r in gemm_kept:
            tile_pairs[(tuple(r["chosen"]["feat"]["tile_vals"]),
                        tuple(r["rejected"]["feat"]["tile_vals"]))] += 1
        print("    most common fast/slow tile tuples:")
        for (a, b), n in tile_pairs.most_common(8):
            print("      %s  ->  %s   x%d" % (a, b, n))

    spreads = sorted(r["spread"] for r in kept)
    if spreads:
        print("  kept spread  median %.2fx  p25 %.2fx  p75 %.2fx"
              % (spreads[len(spreads) // 2],
                 spreads[len(spreads) // 4],
                 spreads[(3 * len(spreads)) // 4]))

    out = args.out or os.path.join(ws, "runs", "glg_speed_pairs.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Strip bulky feat.tiles tuples to keep the jsonl small; keep the bools.
    with open(out, "w") as f:
        for rec in kept:
            slim = dict(rec)
            for side in ("chosen", "rejected"):
                feat = dict(slim[side]["feat"])
                feat.pop("tiles", None)
                slim[side] = {
                    "sample_id": rec[side]["sample_id"],
                    "kernel_ms": rec[side]["kernel_ms"],
                    "path": rec[side]["path"],
                    "n_chars": feat["n_chars"],
                    "tile_vals": list(feat["tile_vals"]),
                    "mma": feat["mma"],
                    "matmul": feat["matmul"],
                    "has_1024": feat["has_1024"],
                    "api": feat["api"],
                }
            f.write(json.dumps(slim) + "\n")
    print("wrote %d pairs to %s" % (len(kept), out))

    # Enough distinct GEMM pairs with a writing difference to bother training.
    writing = sum(
        1 for r in gemm_kept
        if (r["chosen"]["feat"]["tile_vals"] != r["rejected"]["feat"]["tile_vals"]
            or r["chosen"]["feat"]["mma"] != r["rejected"]["feat"]["mma"]
            or r["chosen"]["feat"]["api"] != r["rejected"]["feat"]["api"]))
    print("GEMM pairs with tile/mma/api disagreement: %d" % writing)
    if len(kept) < 80:
        print("VERDICT: too few pairs; do not train")
        return 1
    if gemm < 0.30 * len(band):
        print("VERDICT: GEMM-poor band; do not train")
        return 1
    if writing < 40:
        print("VERDICT: GEMM pairs look like noise, not a writing choice")
        return 1
    print("VERDICT: pairs look teachable; contrastive training is allowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
