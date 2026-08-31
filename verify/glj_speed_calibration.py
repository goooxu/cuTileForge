#!/usr/bin/env python3
"""Calibrate whether the frozen speed dev set reproduces GL-I's matmul signal."""

import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import INCONCLUSIVE_STAGES  # noqa: E402


K = 4


def load_run(prefix):
    path = prefix + "_verified.jsonl"
    if not os.path.isfile(path):
        raise SystemExit("missing %s" % path)
    by = collections.defaultdict(list)
    for line in open(path):
        row = json.loads(line)
        pid, sid = map(int, str(row["key"]).split(":"))
        by[pid].append((sid, row))
    return {pid: [row for _, row in sorted(rows)]
            for pid, rows in by.items()}


def stats(by):
    records = [
        row for rows in by.values() for row in rows
        if row.get("stage") not in INCONCLUSIVE_STAGES
    ]
    passed = sum(bool(row.get("passed")) for row in records)
    solved = sum(any(row.get("passed") for row in rows[:K])
                 for rows in by.values())
    return {
        "n": len(by),
        "solved": solved,
        "p1": 100.0 * passed / len(records) if records else 0.0,
        "p4": 100.0 * solved / len(by) if by else 0.0,
    }


def best_timed(by):
    out = {}
    for pid, rows in by.items():
        candidates = [
            row for row in rows
            if row.get("passed") and row.get("speedup")
            and row.get("kernel_ms")
        ]
        if candidates:
            out[pid] = max(candidates, key=lambda row: row["speedup"])
    return out


def partition(by, categories, category=None):
    return {
        pid: rows for pid, rows in by.items()
        if pid in categories
        and (category is None or categories[pid] == category)
    }


def score(prefix, categories):
    by = load_run(prefix)
    parts = {
        "all": partition(by, categories),
        "matmul": partition(by, categories, "matmul"),
        "conv": partition(by, categories, "conv"),
    }
    return {
        "stats": {name: stats(rows) for name, rows in parts.items()},
        "best": {name: best_timed(rows) for name, rows in parts.items()},
    }


def pairwise(base, candidate):
    common = sorted(set(base) & set(candidate))
    ratios = [
        base[pid]["kernel_ms"] / candidate[pid]["kernel_ms"]
        for pid in common if candidate[pid]["kernel_ms"] > 0
    ]
    return {
        "n": len(ratios),
        "median": statistics.median(ratios) if ratios else 0.0,
        "faster": sum(r >= 1.05 for r in ratios),
        "slower": sum(r <= 0.95 for r in ratios),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    categories = {
        int(row["problem_id"]): row["category"]
        for row in manifest["problems"]
    }
    if len(categories) != 128:
        raise SystemExit("expected 128 frozen tasks, found %d" % len(categories))

    base = score(args.base, categories)
    candidate = score(args.candidate, categories)
    pairs = {
        name: pairwise(base["best"][name], candidate["best"][name])
        for name in ("all", "matmul", "conv")
    }

    print("GL-J speed calibration level 64, k=4")
    for label, row in (("GLE", base), ("GLI", candidate)):
        cells = []
        for name in ("all", "matmul", "conv"):
            stat = row["stats"][name]
            cells.append("%s %d/%d p@1 %.1f%% p@4 %.1f%%"
                         % (name, stat["solved"], stat["n"],
                            stat["p1"], stat["p4"]))
        print("  %-3s %s" % (label, " | ".join(cells)))
    print("kernel_ms vs GLE (>1 GLI faster)")
    for name in ("all", "matmul", "conv"):
        row = pairs[name]
        print("  %-6s %.3fx n=%d  >=1.05 %d  <=0.95 %d"
              % (name, row["median"], row["n"],
                 row["faster"], row["slower"]))

    checks = {
        "matmul_ms": pairs["matmul"]["median"] >= 1.20,
        "matmul_common": pairs["matmul"]["n"] >= 48,
        "conv_solved": (
            candidate["stats"]["conv"]["solved"]
            >= base["stats"]["conv"]["solved"]),
        "conv_p1": (
            candidate["stats"]["conv"]["p1"]
            >= base["stats"]["conv"]["p1"] - 1.0),
        "all_p1": (
            candidate["stats"]["all"]["p1"]
            >= base["stats"]["all"]["p1"] - 1.0),
    }
    passed = all(checks.values())
    print("CALIBRATION %s  %s" % (
        "PASS" if passed else "FAIL",
        " ".join("%s=%s" % (key, "Y" if value else "N")
                 for key, value in checks.items())))
    result = {
        "passed": passed,
        "checks": checks,
        "base": base["stats"],
        "candidate": candidate["stats"],
        "kernel_ms": pairs,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
