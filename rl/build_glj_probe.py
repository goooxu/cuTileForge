#!/usr/bin/env python3
"""Build the frozen non-eval probe used to choose the GL-I delta scale.

The probe is deliberately outside level 60 and every sealed held-out level.
It samples equal numbers of matmul and conv tasks from the four levels used to
harvest GL-I's training data, while excluding every task listed in GL-I's ORPO
and retain inputs.  Selected source files are copied and renumbered into
KernelBench level 62; the manifest makes the selection auditable.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


SOURCE_LEVELS = (86, 87, 92, 93)
CATEGORIES = ("matmul", "conv")
CATEGORY_RE = re.compile(r"\(tier\s+\d+,\s*([a-z_]+)\)")
FILE_RE = re.compile(r"^(\d+)_.*\.py$")


def load_jsonl(path: Path):
    if not path.is_file():
        raise SystemExit("missing training data: %s" % path)
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def training_keys(orpo, retain):
    keys = set()
    for path in (orpo, retain):
        for row in load_jsonl(path):
            keys.add((int(row["level"]), int(row["problem_id"])))
    return keys


def category_of(path):
    text = path.read_text()
    match = CATEGORY_RE.search(text)
    return match.group(1) if match else None


def stable_order(seed, level, pid, name):
    raw = "%d:%d:%d:%s" % (seed, level, pid, name)
    return hashlib.sha256(raw.encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--level", type=int, default=62)
    ap.add_argument("--per-category", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    forge = Path(__file__).resolve().parents[1]
    kb_root = forge / "kernelbench" / "KernelBench"
    runs = workspace / "runs"
    excluded = training_keys(
        runs / "orpo_gli.jsonl", runs / "sft_gli_retain.jsonl")

    # Round-robin quotas keep all four source levels represented.  The stable
    # hash, not directory order, chooses tasks within a level.
    base = args.per_category // len(SOURCE_LEVELS)
    extra = args.per_category % len(SOURCE_LEVELS)
    quotas = {
        (category, level): base + (i < extra)
        for category in CATEGORIES
        for i, level in enumerate(SOURCE_LEVELS)
    }

    candidates = defaultdict(list)
    excluded_by_category = defaultdict(int)
    for level in SOURCE_LEVELS:
        source = kb_root / ("level%d" % level)
        if not source.is_dir():
            raise SystemExit("missing source level: %s" % source)
        for path in source.glob("*.py"):
            match = FILE_RE.match(path.name)
            if not match:
                continue
            pid = int(match.group(1))
            category = category_of(path)
            if category not in CATEGORIES:
                continue
            if (level, pid) in excluded:
                excluded_by_category[category] += 1
                continue
            order = stable_order(args.seed, level, pid, path.name)
            candidates[(category, level)].append((order, pid, path))

    selected = []
    for category in CATEGORIES:
        for level in SOURCE_LEVELS:
            rows = sorted(candidates[(category, level)])
            quota = quotas[(category, level)]
            if len(rows) < quota:
                raise SystemExit(
                    "%s level %d has %d candidates, need %d"
                    % (category, level, len(rows), quota))
            for order, pid, path in rows[:quota]:
                selected.append({
                    "category": category,
                    "source_level": level,
                    "source_problem_id": pid,
                    "source_file": path.name,
                    "selection_hash": order,
                    "path": path,
                })

    selected.sort(key=lambda row: (
        CATEGORIES.index(row["category"]),
        row["source_level"],
        row["selection_hash"],
    ))

    out_dir = kb_root / ("level%d" % args.level)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    problems = []
    for new_id, row in enumerate(selected, 1):
        out_name = "%d_%s" % (new_id, row["source_file"].split("_", 1)[1])
        out_path = out_dir / out_name
        shutil.copy2(row["path"], out_path)
        problems.append({
            "problem_id": new_id,
            "file": out_name,
            "category": row["category"],
            "source_level": row["source_level"],
            "source_problem_id": row["source_problem_id"],
            "source_file": row["source_file"],
            "source_sha256": sha256(row["path"]),
            "selection_hash": row["selection_hash"],
        })

    manifest = {
        "probe_level": args.level,
        "seed": args.seed,
        "source_levels": list(SOURCE_LEVELS),
        "categories": list(CATEGORIES),
        "per_category": args.per_category,
        "excluded_training_tasks": len(excluded),
        "excluded_by_category": dict(sorted(excluded_by_category.items())),
        "problems": problems,
    }
    manifest_path = runs / "glj_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = sha256(manifest_path)
    counts = defaultdict(int)
    levels = defaultdict(int)
    for row in problems:
        counts[row["category"]] += 1
        levels[(row["category"], row["source_level"])] += 1
    print("wrote %d problems to %s" % (len(problems), out_dir))
    print("manifest %s sha256 %s" % (manifest_path, digest))
    print("categories %s" % dict(sorted(counts.items())))
    print("category/level %s" % {
        "%s/%d" % key: value for key, value in sorted(levels.items())
    })
    print("excluded training keys %d; by category %s"
          % (len(excluded), dict(sorted(excluded_by_category.items()))))


if __name__ == "__main__":
    main()
