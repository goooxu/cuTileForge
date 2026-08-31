#!/usr/bin/env python3
"""Build a fresh, hash-disjoint matmul/conv pool for GL-J speed calibration.

The stock task generator has a deliberately small shape ladder.  A new seed
therefore exhausts after about a hundred matmul tasks and repeats shapes seen by
training.  This builder takes those independently sampled graphs and expands
them over a separate, non-power-of-two shape grid.  Conv graphs are made unique
by changing only batch size, preserving all channel/group constraints.

No model output is consulted here.  The resulting level 63 pool is frozen
before GL-E harvest decides which tasks are actually tile-sensitive.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "taskgen"))
from generate_tasks import task_hash  # noqa: E402


CATEGORY_RE = re.compile(r"\(tier\s+\d+,\s*([a-z_]+)\)")
CONST_RE = r"(?m)^%s\s*=\s*\d+\s*$"

# Separate from every taskgen shape ladder.  Sizes are large enough that tile
# choice matters but stay below roughly 200 MB per tensor.
MATMUL_SHAPES = (
    (576, 960, 704),
    (640, 896, 768),
    (704, 1088, 832),
    (768, 1152, 896),
    (832, 1216, 960),
    (896, 1280, 1024),
    (960, 1408, 1088),
    (1024, 1536, 1152),
    (1088, 1664, 1216),
    (1152, 1792, 1280),
    (1216, 1920, 1408),
    (1280, 2048, 1536),
    (1408, 2176, 1664),
    (1536, 2304, 1792),
    (1664, 2432, 1920),
    (1792, 2560, 2048),
    (1920, 2816, 2176),
    (2048, 3072, 2304),
    (2176, 3328, 2432),
    (2304, 3584, 2560),
    (2432, 3840, 2816),
    (2560, 4096, 3072),
    (2816, 4352, 3328),
    (3072, 4608, 3584),
)
BMM_SHAPES = (
    (3, 128, 192, 160),
    (4, 144, 224, 176),
    (5, 160, 256, 192),
    (8, 176, 288, 208),
    (6, 192, 320, 224),
    (9, 208, 352, 240),
    (7, 224, 384, 256),
    (10, 240, 416, 288),
    (3, 256, 448, 320),
    (4, 288, 480, 352),
    (5, 320, 512, 384),
    (8, 352, 576, 416),
    (6, 384, 640, 448),
    (9, 416, 704, 480),
    (7, 448, 768, 512),
    (10, 480, 832, 576),
)
CONV_BATCHES = (3, 5, 6, 7, 9, 10, 12, 14)


def category_of(source):
    match = CATEGORY_RE.search(source)
    return match.group(1) if match else None


def replace_int(source, name, value):
    updated, n = re.subn(CONST_RE % re.escape(name),
                         "%s = %d" % (name, value), source)
    if n != 1:
        raise ValueError("expected one %s constant, found %d" % (name, n))
    return updated


def stable_index(seed, name, variant, size):
    raw = "%d:%s:%d" % (seed, name, variant)
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16) % size


def hashes_of_other_levels(kb_root, target_level):
    hashes = set()
    for path in sorted(kb_root.glob("level*")):
        if path.name == "level%d" % target_level or not path.is_dir():
            continue
        for task in path.glob("*.py"):
            hashes.add(task_hash(task.read_text()))
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--level", type=int, default=63)
    ap.add_argument("--matmul-variants", type=int, default=4)
    ap.add_argument("--conv-count", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    forge = Path(__file__).resolve().parents[1]
    kb_root = forge / "kernelbench" / "KernelBench"
    raw_matmul = workspace / "runs" / "glj_speed_raw" / "level1"
    raw_conv = workspace / "runs" / "glj_speed_raw" / "level2"
    for path in (raw_matmul, raw_conv):
        if not path.is_dir():
            raise SystemExit("missing raw task pool: %s" % path)

    excluded = hashes_of_other_levels(kb_root, args.level)
    seen = set(excluded)
    rows = []

    matmul_files = []
    for path in sorted(raw_matmul.glob("*.py")):
        source = path.read_text()
        if category_of(source) == "matmul":
            matmul_files.append((path, source))

    for path, source in matmul_files:
        is_bmm = re.search(CONST_RE % "batch_size", source) is not None
        shapes = BMM_SHAPES if is_bmm else MATMUL_SHAPES
        for variant in range(args.matmul_variants):
            shape = shapes[stable_index(args.seed, path.name, variant, len(shapes))]
            updated = source
            if is_bmm:
                batch, m, k, n = shape
                updated = replace_int(updated, "batch_size", batch)
            else:
                m, k, n = shape
            updated = replace_int(updated, "M", m)
            updated = replace_int(updated, "K", k)
            updated = replace_int(updated, "N", n)
            digest = task_hash(updated)
            if digest in seen:
                continue
            seen.add(digest)
            rows.append({
                "category": "matmul",
                "source_file": path.name,
                "variant": variant,
                "shape": list(shape),
                "task_hash": digest,
                "source": updated,
            })

    conv_candidates = []
    for path in sorted(raw_conv.glob("*.py")):
        source = path.read_text()
        if category_of(source) != "conv":
            continue
        if re.search(CONST_RE % "batch_size", source) is None:
            continue
        order = hashlib.sha256(
            ("%d:%s" % (args.seed, path.name)).encode()).hexdigest()
        conv_candidates.append((order, path, source))
    conv_candidates.sort()

    for i, (_, path, source) in enumerate(conv_candidates):
        if sum(row["category"] == "conv" for row in rows) >= args.conv_count:
            break
        batch = CONV_BATCHES[i % len(CONV_BATCHES)]
        updated = replace_int(source, "batch_size", batch)
        digest = task_hash(updated)
        if digest in seen:
            continue
        seen.add(digest)
        rows.append({
            "category": "conv",
            "source_file": path.name,
            "variant": 0,
            "shape": {"batch_size": batch},
            "task_hash": digest,
            "source": updated,
        })

    counts = Counter(row["category"] for row in rows)
    if counts["matmul"] < 300:
        raise SystemExit("only %d unique matmul tasks; need at least 300"
                         % counts["matmul"])
    if counts["conv"] < 64:
        raise SystemExit("only %d unique conv tasks; need at least 64"
                         % counts["conv"])

    out_dir = kb_root / ("level%d" % args.level)
    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(parents=True)

    manifest_rows = []
    rows.sort(key=lambda row: (
        0 if row["category"] == "matmul" else 1,
        row["task_hash"],
    ))
    for pid, row in enumerate(rows, 1):
        stem = row["source_file"].split("_", 1)[1]
        filename = "%d_%s" % (pid, stem)
        (out_dir / filename).write_text(row.pop("source"))
        manifest_rows.append(dict(row, problem_id=pid, file=filename))

    manifest = {
        "pool_level": args.level,
        "seed": args.seed,
        "matmul_variants": args.matmul_variants,
        "counts": dict(sorted(counts.items())),
        "excluded_hashes": len(excluded),
        "problems": manifest_rows,
    }
    manifest_path = workspace / "runs" / "glj_speed_pool_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print("wrote %d tasks to %s: %s" % (
        len(rows), out_dir, dict(sorted(counts.items()))))
    print("excluded %d existing task hashes" % len(excluded))
    print("manifest %s sha256 %s" % (manifest_path, manifest_sha))


if __name__ == "__main__":
    main()
