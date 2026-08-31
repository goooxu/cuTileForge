#!/usr/bin/env python3
"""Freeze 16 matmul + 16 conv level-64 tasks for six-model screening."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--source-level", type=int, default=64)
    ap.add_argument("--out-level", type=int, default=65)
    ap.add_argument("--per-category", type=int, default=16)
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    forge = Path(__file__).resolve().parents[1]
    kb_root = forge / "kernelbench" / "KernelBench"
    source_manifest = workspace / "runs" / "glj_speed_dev_manifest.json"
    if not source_manifest.is_file():
        raise SystemExit("missing frozen level-64 manifest: %s"
                         % source_manifest)
    manifest = json.loads(source_manifest.read_text())
    source_by_id = {
        int(row["problem_id"]): row for row in manifest["problems"]
    }

    chosen = []
    for category in ("matmul", "conv"):
        rows = sorted(
            [row for row in source_by_id.values()
             if row["category"] == category],
            key=lambda row: row["task_hash"],
        )
        if len(rows) < args.per_category:
            raise SystemExit("%s has %d tasks, need %d"
                             % (category, len(rows), args.per_category))
        chosen.extend(rows[:args.per_category])

    source_dir = kb_root / ("level%d" % args.source_level)
    out_dir = kb_root / ("level%d" % args.out_level)
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(parents=True)

    frozen = []
    for pid, row in enumerate(chosen, 1):
        source_path = source_dir / row["file"]
        stem = row["file"].split("_", 1)[1]
        filename = "%d_%s" % (pid, stem)
        dest = out_dir / filename
        shutil.copy2(str(source_path), str(dest))
        frozen.append({
            "problem_id": pid,
            "file": filename,
            "category": row["category"],
            "task_hash": row["task_hash"],
            "source_level": args.source_level,
            "source_problem_id": row["problem_id"],
            "source_file": row["file"],
            "source_sha256": file_sha(source_path),
        })

    output = {
        "level": args.out_level,
        "source_manifest_sha256": file_sha(source_manifest),
        "selection": {
            "categories": ["matmul", "conv"],
            "per_category": args.per_category,
            "order": "task_hash ascending",
        },
        "problems": frozen,
    }
    out_manifest = workspace / "runs" / "sequential_screen_manifest.json"
    out_manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("wrote %d tasks to %s" % (len(frozen), out_dir))
    print("manifest %s sha256 %s"
          % (out_manifest, file_sha(out_manifest)))


if __name__ == "__main__":
    main()
