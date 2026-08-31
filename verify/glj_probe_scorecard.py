#!/usr/bin/env python3
"""Score GL-I delta scales on the frozen level-62 probe.

Selection is intentionally conservative and fixed before any candidate runs:

* matmul median kernel_ms ratio versus GL-E >= 1.30x;
* matmul, conv, and all-task solved@4 cannot fall below GL-E;
* conv and all-task pass@1 may fall by at most 1 percentage point.

Candidates are supplied in increasing scale order.  The smallest scale passing
all gates is selected, preserving the most GL-E behavior while retaining a
clear margin over table A's 1.20x matmul requirement.
"""

import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import INCONCLUSIVE_STAGES  # noqa: E402


K = 4
MIN_MATMUL_MS = 1.30
MAX_P1_DROP = 1.0


def load_manifest(path):
    data = json.load(open(path))
    return data, {int(row["problem_id"]): row for row in data["problems"]}


def load_run(prefix):
    path = prefix + "_verified.jsonl"
    if not os.path.isfile(path):
        raise SystemExit("missing %s" % path)
    by = collections.defaultdict(list)
    for line in open(path):
        row = json.loads(line)
        pid, sid = map(int, str(row["key"]).split(":"))
        by[pid].append((sid, row))
    return {
        pid: [row for _, row in sorted(rows)]
        for pid, rows in by.items()
    }


def subset(by, manifest_by, category=None):
    return {
        pid: rows for pid, rows in by.items()
        if pid in manifest_by
        and (category is None or manifest_by[pid]["category"] == category)
    }


def pass_stats(by):
    n = len(by)
    records = [
        row for rows in by.values() for row in rows
        if row.get("stage") not in INCONCLUSIVE_STAGES
    ]
    passed = sum(bool(row.get("passed")) for row in records)
    solved = sum(
        any(row.get("passed") for row in rows[:K])
        for rows in by.values()
    )
    return {
        "n": n,
        "solved": solved,
        "p1": 100.0 * passed / len(records) if records else 0.0,
        "p4": 100.0 * solved / n if n else 0.0,
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


def pairwise_ms(base, candidate):
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


def score(tag, prefix, manifest_by):
    by = load_run(prefix)
    parts = {
        "all": subset(by, manifest_by),
        "matmul": subset(by, manifest_by, "matmul"),
        "conv": subset(by, manifest_by, "conv"),
    }
    return {
        "tag": tag,
        "prefix": prefix,
        "stats": {name: pass_stats(rows) for name, rows in parts.items()},
        "best": {name: best_timed(rows) for name, rows in parts.items()},
    }


def print_score(row):
    cells = []
    for name in ("all", "matmul", "conv"):
        stat = row["stats"][name]
        cells.append(
            "%s %d/%d p@1 %.1f%%"
            % (name, stat["solved"], stat["n"], stat["p1"]))
    print("  %-8s %s" % (row["tag"], " | ".join(cells)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True, metavar="TAG:PREFIX")
    ap.add_argument("--candidate", action="append", default=[],
                    metavar="TAG:SCALE:PREFIX")
    ap.add_argument("--control", action="append", default=[],
                    metavar="TAG:PREFIX")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    manifest, manifest_by = load_manifest(args.manifest)
    if len(manifest_by) != 64:
        raise SystemExit("probe manifest has %d problems, expected 64"
                         % len(manifest_by))

    base_tag, sep, base_prefix = args.base.partition(":")
    if not sep:
        raise SystemExit("--base expects TAG:PREFIX")
    base = score(base_tag, base_prefix, manifest_by)

    candidates = []
    for spec in args.candidate:
        tag, scale, prefix = spec.split(":", 2)
        row = score(tag, prefix, manifest_by)
        row["scale"] = float(scale)
        candidates.append(row)
    controls = []
    for spec in args.control:
        tag, sep, prefix = spec.partition(":")
        if not sep:
            raise SystemExit("--control expects TAG:PREFIX")
        controls.append(score(tag, prefix, manifest_by))

    print("GL-J scale probe level %d, k=%d, manifest seed %s"
          % (manifest["probe_level"], K, manifest["seed"]))
    print_score(base)
    for row in candidates + controls:
        print_score(row)

    summaries = []
    print()
    print("pairwise kernel_ms vs %s (>1 candidate faster)" % base_tag)
    for row in candidates + controls:
        pairs = {
            name: pairwise_ms(base["best"][name], row["best"][name])
            for name in ("all", "matmul", "conv")
        }
        for name in ("all", "matmul", "conv"):
            pair = pairs[name]
            print("  %-8s %-6s %.3fx n=%d  >=1.05 %d  <=0.95 %d"
                  % (row["tag"], name, pair["median"], pair["n"],
                     pair["faster"], pair["slower"]))
        summaries.append((row, pairs))

    selected = None
    decisions = []
    for row, pairs in summaries[:len(candidates)]:
        checks = {
            "matmul_ms": pairs["matmul"]["median"] >= MIN_MATMUL_MS,
            "matmul_solved": (
                row["stats"]["matmul"]["solved"]
                >= base["stats"]["matmul"]["solved"]),
            "conv_solved": (
                row["stats"]["conv"]["solved"]
                >= base["stats"]["conv"]["solved"]),
            "all_solved": (
                row["stats"]["all"]["solved"]
                >= base["stats"]["all"]["solved"]),
            "conv_p1": (
                row["stats"]["conv"]["p1"]
                >= base["stats"]["conv"]["p1"] - MAX_P1_DROP),
            "all_p1": (
                row["stats"]["all"]["p1"]
                >= base["stats"]["all"]["p1"] - MAX_P1_DROP),
        }
        passed = all(checks.values())
        decisions.append({
            "tag": row["tag"],
            "scale": row["scale"],
            "passed": passed,
            "checks": checks,
            "matmul_kernel_ms": pairs["matmul"],
            "stats": row["stats"],
        })
        print("gate %-8s scale %.2f: %s  %s"
              % (row["tag"], row["scale"], "PASS" if passed else "FAIL",
                 " ".join("%s=%s" % (key, "Y" if value else "N")
                          for key, value in checks.items())))
        if selected is None and passed:
            selected = row

    if selected:
        print("SELECT %s scale=%.2f" % (selected["tag"], selected["scale"]))
    else:
        print("SELECT none")

    result = {
        "base": {
            "tag": base["tag"],
            "stats": base["stats"],
        },
        "gates": {
            "min_matmul_kernel_ms": MIN_MATMUL_MS,
            "max_p1_drop_points": MAX_P1_DROP,
            "coverage_not_below_base": True,
            "selection": "smallest passing scale",
        },
        "candidates": decisions,
        "selected": (
            {"tag": selected["tag"], "scale": selected["scale"]}
            if selected else None),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
