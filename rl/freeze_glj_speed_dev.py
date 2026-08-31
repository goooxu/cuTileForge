#!/usr/bin/env python3
"""Freeze tile-sensitive level-63 tasks into the GL-J calibration level 64."""

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified(path):
    by = collections.defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = int(str(row["key"]).split(":")[0])
            by[pid].append(row)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--pool-manifest", default=None)
    ap.add_argument("--verified", default=None)
    ap.add_argument("--source-level", type=int, default=63)
    ap.add_argument("--out-level", type=int, default=64)
    ap.add_argument("--per-category", type=int, default=64)
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    forge = Path(__file__).resolve().parents[1]
    kb_root = forge / "kernelbench" / "KernelBench"
    manifest_path = Path(args.pool_manifest or
                         workspace / "runs" / "glj_speed_pool_manifest.json")
    verified_path = Path(args.verified or
                         workspace / "runs" / "harvest_glj63_gle_verified.jsonl")
    manifest = json.loads(manifest_path.read_text())
    manifest_by = {
        int(row["problem_id"]): row for row in manifest["problems"]
    }
    verified = load_verified(verified_path)

    eligible_matmul = []
    eligible_conv = []
    for pid, meta in manifest_by.items():
        rows = verified.get(pid, [])
        passed = [row for row in rows if row.get("passed")]
        timed = [
            row for row in passed
            if row.get("kernel_ms") and row.get("speedup")
            and 0.05 <= float(row["kernel_ms"]) <= 50.0
        ]
        base = {
            "source_problem_id": pid,
            "category": meta["category"],
            "source_file": meta["file"],
            "task_hash": meta["task_hash"],
            "n_passed": len(passed),
            "n_timed_in_range": len(timed),
        }
        if meta["category"] == "matmul" and len(timed) >= 2:
            fastest = min(float(row["kernel_ms"]) for row in timed)
            slowest = max(float(row["kernel_ms"]) for row in timed)
            spread = slowest / fastest
            if 1.5 <= spread <= 8.0:
                eligible_matmul.append(dict(
                    base, min_kernel_ms=fastest,
                    max_kernel_ms=slowest, spread=spread))
        elif meta["category"] == "conv" and passed:
            eligible_conv.append(base)

    eligible_matmul.sort(key=lambda row: row["task_hash"])
    eligible_conv.sort(key=lambda row: row["task_hash"])
    print("eligible matmul %d / %d; conv %d / %d" % (
        len(eligible_matmul),
        sum(row["category"] == "matmul" for row in manifest_by.values()),
        len(eligible_conv),
        sum(row["category"] == "conv" for row in manifest_by.values()),
    ))
    if len(eligible_matmul) < args.per_category:
        raise SystemExit("need %d tile-sensitive matmul tasks, found %d"
                         % (args.per_category, len(eligible_matmul)))
    if len(eligible_conv) < args.per_category:
        raise SystemExit("need %d solved conv tasks, found %d"
                         % (args.per_category, len(eligible_conv)))

    chosen = (
        eligible_matmul[:args.per_category]
        + eligible_conv[:args.per_category])
    source_dir = kb_root / ("level%d" % args.source_level)
    out_dir = kb_root / ("level%d" % args.out_level)
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(parents=True)

    frozen = []
    for new_pid, row in enumerate(chosen, 1):
        stem = row["source_file"].split("_", 1)[1]
        filename = "%d_%s" % (new_pid, stem)
        shutil.copy2(str(source_dir / row["source_file"]),
                     str(out_dir / filename))
        frozen.append(dict(row, problem_id=new_pid, file=filename))

    frozen_manifest = {
        "level": args.out_level,
        "source_level": args.source_level,
        "selection": {
            "per_category": args.per_category,
            "matmul_min_timed": 2,
            "kernel_ms_range": [0.05, 50.0],
            "spread_range": [1.5, 8.0],
            "ordering": "task_hash ascending",
        },
        "pool_manifest_sha256": sha256(manifest_path),
        "harvest_verified": str(verified_path),
        "eligible_counts": {
            "matmul": len(eligible_matmul),
            "conv": len(eligible_conv),
        },
        "problems": frozen,
    }
    out_manifest = workspace / "runs" / "glj_speed_dev_manifest.json"
    out_manifest.write_text(
        json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")
    print("wrote %d tasks to %s" % (len(frozen), out_dir))
    print("manifest %s sha256 %s" % (out_manifest, sha256(out_manifest)))


if __name__ == "__main__":
    main()
