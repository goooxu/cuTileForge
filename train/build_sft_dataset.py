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


def category_of(ref_src: str) -> str:
    """Operator family, read from the docstring the task generator writes."""
    m = re.search(r'"""(\w+) \(tier (\d+), (\w+)\)', ref_src)
    return m.group(3) if m else "?"


def apply_quota(records, quota: dict):
    """Cap each category, spending the budget on breadth of tasks first.

    Without this the mix follows how easy each family is to sample rather than
    where the model needs help: matmul and elementwise pass 80-100% of the time
    and would take well over half the dataset, yet both are already at their
    ceiling on the benchmark -- feeding ~170 matmul examples in the previous
    round moved it -0.6pp. Convolution and pooling are the opposite.

    Selection is round-robin over distinct problems, so a category that gets cut
    loses repeated solutions to the same task before it loses coverage.
    """
    by_cat = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in records:
        # Problem ids restart per level, so a task is only unique with its level.
        by_cat[rec["category"]][(rec["level"], rec["problem_id"])].append(rec)

    kept = []
    for cat, by_problem in by_cat.items():
        limit = quota.get(cat, quota.get("*", None))
        flat = []
        # Round-robin: one solution from each problem, then a second, and so on.
        for i in range(max(len(v) for v in by_problem.values())):
            for pid in sorted(by_problem):
                if i < len(by_problem[pid]):
                    flat.append(by_problem[pid][i])
        kept.extend(flat if limit is None else flat[:limit])
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LEVEL:KERNEL_DIR:VERIFIED",
                    help="Repeatable. Each is a synthetic run to draw solutions from.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-problem", type=int, default=3,
                    help="Cap solutions kept per task so easy tasks do not dominate.")
    ap.add_argument("--prompt-tier", default="cutile_docs",
                    help="Which prompt composition to train on. The full "
                         "reference is 91%% of every sequence, so cutile_concepts "
                         "(6.2x shorter) or cutile_nodocs (15.9x) is what makes a "
                         "large dataset affordable. Generation still uses the full "
                         "docs; this only changes what the model is trained to "
                         "condition on.")
    ap.add_argument("--nodocs-fraction", type=float, default=0.0,
                    help="Share of examples rendered with no documentation at "
                         "all, mixed into the tier above. Writing the DSL from "
                         "memory is the goal, and the model already manages it "
                         "5.5%% of the time untrained.")
    ap.add_argument("--min-speedup", type=float, default=None,
                    help="Drop solutions slower than this multiple of the torch "
                         "reference. Samples with no timing are kept.")
    ap.add_argument("--category-quota", default=None,
                    help="Comma-separated cat=N caps, e.g. matmul=100,elementwise=60. "
                         "Use '*=N' for a default. Uncapped categories keep everything.")
    args = ap.parse_args()

    quota = {}
    if args.category_quota:
        for pair in args.category_quota.split(","):
            cat, _, n = pair.partition("=")
            quota[cat.strip()] = int(n)

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
                passed.append((int(pid), int(sid), r.get("speedup")))
        print("level %d: %d verified passing samples" % (level, len(passed)))

        n_slow = 0
        if args.min_speedup is not None:
            before = len(passed)
            # A kernel with no timing cannot be judged, so keep it: correctness
            # sets that were verified without --measure-time would otherwise
            # vanish entirely.
            passed = [p for p in passed
                      if p[2] is None or p[2] >= args.min_speedup]
            n_slow = before - len(passed)
            print("  dropped %d below %.2fx" % (n_slow, args.min_speedup))

        by_problem = collections.defaultdict(list)
        for pid, sid, sp in passed:
            by_problem[pid].append((sid, sp))
        # Fastest first, so the per-problem cap keeps the best solutions rather
        # than whichever the sampler happened to emit first. Among several
        # correct kernels for one task, the difference between them is the most
        # direct signal about speed the dataset can carry.
        for pid in by_problem:
            by_problem[pid].sort(key=lambda t: (-(t[1] or 0.0), t[0]))
        n_tasks += len(by_problem)

        for pid, sids in sorted(by_problem.items()):
            problem = dataset.get_problem_by_id(pid)
            # Deterministic per task, so the same task always lands in the same
            # tier and the two do not disagree about what it looks like.
            tier = args.prompt_tier
            if args.nodocs_fraction > 0:
                h = int(hashlib.sha256(problem.name.encode()).hexdigest()[:8], 16)
                if (h % 1000) / 1000.0 < args.nodocs_fraction:
                    tier = "cutile_nodocs"
            prompt = get_custom_prompt(
                tier,
                ref_arch_src=problem.code,
                backend="cutile",
                option="one_shot",
                precision="fp32",
            )
            n_for_problem = 0
            for sid, speedup in sids:
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
                    "category": category_of(problem.code),
                    "speedup": speedup,
                    "prompt_tier": tier,
                    "prompt": prompt,
                    # Fenced so the target matches the format the model is asked
                    # for and the eval-time extractor expects.
                    "completion": "```python\n%s\n```" % code.strip(),
                })

    before = collections.Counter(r["category"] for r in kept)
    if quota:
        kept = apply_quota(kept, quota)
    after = collections.Counter(r["category"] for r in kept)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    print("kept %d examples from %d distinct tasks" % (len(kept), n_tasks))
    print("  dropped %d duplicates, %d over per-task cap" % (dropped_dup, dropped_cap))

    print("  by category:")
    for cat in sorted(before, key=lambda c: -before[c]):
        n_after = after.get(cat, 0)
        note = "" if n_after == before[cat] else "  (capped from %d)" % before[cat]
        print("    %-12s %4d  %4.1f%%%s"
              % (cat, n_after, n_after / max(len(kept), 1) * 100, note))

    tasks = {(r["level"], r["problem_id"]) for r in kept}
    print("  distinct tasks covered: %d" % len(tasks))

    sp = sorted(r["speedup"] for r in kept if r.get("speedup"))
    if sp:
        print("  speedup of kept solutions: median %.2fx, %d/%d beat torch"
              % (sp[len(sp) // 2], sum(1 for s in sp if s > 1.0), len(sp)))

    tiers = collections.Counter(r["prompt_tier"] for r in kept)
    chars = sum(len(r["prompt"]) + len(r["completion"]) for r in kept)
    print("  prompt tiers: %s" % dict(tiers))
    print("  total %0.1fM characters, ~%0.1fM tokens per epoch"
          % (chars / 1e6, chars / 4 / 1e6))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
