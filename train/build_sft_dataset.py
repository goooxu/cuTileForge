"""Turn verified cuTile kernels into an SFT dataset.

Only samples the verifier accepted are used, which is what makes this training
corpus self-generated rather than external: the cuTile compiler and a numerical
comparison against PyTorch decide what is correct, with no human-written
reference solutions involved.

The prompt is rendered with exactly the composition used at evaluation time
(backend=cutile, custom_prompt_key=cutile_docs), so training and inference see
the same format and the held-out comparison stays honest.

Usage:
    python3 train/build_sft_dataset.py \\
        --verified runs/synth_l92_verified.jsonl \\
        --kernel-dir runs/synth_l92 --level 92 \\
        --out runs/sft_l92.jsonl
"""

import argparse
import collections
import hashlib
import json
import os
import re


def normalise(code: str) -> str:
    """Canonical form for dedup: ignore comments, blank lines and indentation width."""
    out = []
    for line in code.splitlines():
        line = re.sub(r"#.*$", "", line).rstrip()
        if line.strip():
            out.append(re.sub(r"\s+", " ", line.strip()))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LEVEL:KERNEL_DIR:VERIFIED",
                    help="Repeatable. Each is a synthetic run to draw solutions from.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=3,
                    help="Cap solutions kept per task so easy tasks do not dominate.")
    args = ap.parse_args()

    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    kept, seen_hashes = [], set()
    dropped_dup = dropped_cap = 0
    n_tasks = 0

    for spec in args.run:
        level_s, kernel_dir, verified = spec.split(":", 2)
        level = int(level_s)
        dataset = construct_kernelbench_dataset(level)

        passed = []
        for line in open(verified):
            r = json.loads(line)
            if r.get("passed"):
                pid, sid = r["key"].split(":")
                passed.append((int(pid), int(sid)))
        print("level %d: %d verified passing samples" % (level, len(passed)))

        by_problem = collections.defaultdict(list)
        for pid, sid in sorted(passed):
            by_problem[pid].append(sid)
        n_tasks += len(by_problem)

        for pid, sids in sorted(by_problem.items()):
            problem = dataset.get_problem_by_id(pid)
            prompt = get_custom_prompt(
                "cutile_docs",
                ref_arch_src=problem.code,
                backend="cutile",
                option="one_shot",
                precision="fp32",
            )
            n_for_problem = 0
            for sid in sids:
                if n_for_problem >= args.max_per_problem:
                    dropped_cap += 1
                    continue
                path = os.path.join(
                    kernel_dir,
                    "level_%d_problem_%d_sample_%d_kernel.py" % (level, pid, sid))
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    code = f.read()

                h = hashlib.sha256(normalise(code).encode()).hexdigest()
                if h in seen_hashes:
                    dropped_dup += 1
                    continue
                seen_hashes.add(h)
                n_for_problem += 1

                kept.append({
                    "level": level,
                    "problem_id": pid,
                    "sample_id": sid,
                    "problem": problem.name,
                    "prompt": prompt,
                    # Fenced so the target matches the format the model is asked
                    # for and the eval-time extractor expects.
                    "completion": "```python\n%s\n```" % code.strip(),
                })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    print("kept %d examples from %d distinct tasks" % (len(kept), n_tasks))
    print("  dropped %d duplicates, %d over per-task cap" % (dropped_dup, dropped_cap))

    cats = collections.Counter()
    for rec in kept:
        m = re.match(r"\d+_(.+?)_t\d+", rec["problem"].replace(".py", ""))
        cats[m.group(1) if m else "?"] += 1
    print("  top operators:", dict(cats.most_common(12)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
