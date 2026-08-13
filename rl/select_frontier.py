"""Pick the tasks GRPO can actually learn from.

Group-relative advantage is the reward minus the group's mean, so a task the
policy always fails and a task it always solves both contribute exactly zero.
Spending rollouts on either is spending them for nothing, and at pass@1 near 14%
the always-fails group is most of the task set.

So screen first: sample k per task with the current policy, keep the ones whose
pass rate is strictly between 0 and 1. Screening is cheap next to training --
a few thousand rollouts and one pass through the verifier -- and it decides how
much of every later iteration does any work.

Usage:
    python3 rl/select_frontier.py --levels 90,91,92,93,94 --samples 6 \\
        --out runs/rl_frontier.json
"""

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "repair"))
sys.path.insert(0, os.path.join(HERE, "..", "verify"))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, HERE)

from build_sft_dataset import category_of  # noqa: E402
from repair_loop import Chat, extract_code, sample_batch  # noqa: E402
from reward import reward_for  # noqa: E402
from worker import VerifierPool  # noqa: E402


def apply_category_quota(frontier, spec: str):
    """Cap each operator family's share of the frontier.

    Sorting by reward spread alone lets the most numerous family take the
    rollouts. On the run that stacked GRPO onto the distilled model, convolution
    and pooling were 65% of the frontier while activation had no builder at all,
    and the result gained 19 problems overall but lost 3 on matmul and 2 on
    activation -- families the policy had been good at and then drifted away
    from. Entries stay in spread order within each family, so the cap trims the
    least informative tasks first.

    spec is "cat=N" pairs, with "*=N" as the default for unlisted families.
    """
    caps = {}
    for pair in spec.split(","):
        cat, _, n = pair.partition("=")
        caps[cat.strip()] = int(n)
    default = caps.pop("*", None)

    kept, seen = [], collections.Counter()
    for e in frontier:
        c = e.get("category", "?")
        cap = caps.get(c, default)
        if cap is not None and seen[c] >= cap:
            continue
        seen[c] += 1
        kept.append(e)
    print("  category quota: %d -> %d tasks, %s"
          % (len(frontier), len(kept), dict(seen.most_common())))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True, help="Comma-separated, e.g. 90,91,92")
    ap.add_argument("--samples", type=int, default=6,
                    help="Rollouts per task for the screen.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen3-Coder-Next")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--verify-workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--limit-per-level", type=int, default=None)
    ap.add_argument("--prompt-tier", default="cutile_docs",
                    help="Must match the tier training will sample at, or the "
                         "pass rates screened here describe a different policy.")
    ap.add_argument("--mode", default="learnable", choices=["learnable", "solid"],
                    help="learnable keeps tasks the policy sometimes solves, for "
                         "improving correctness. solid keeps the ones it always "
                         "solves, for improving speed: correctness is then "
                         "constant within a group and contributes no advantage, "
                         "so the speed term carries the whole gradient.")
    ap.add_argument("--category-quota", default=None,
                    help="Comma-separated cat=N caps, '*=N' for the rest. "
                         "Without this the frontier's composition is whatever "
                         "the task pool happened to contain.")
    ap.add_argument("--from-run", default=None,
                    help="Skip sampling and derive the frontier from an existing "
                         "verified run (JSONL from verify/fast_verify.py). A "
                         "k=16 harvest already measures each task's pass rate "
                         "far better than a k=6 screen would.")
    args = ap.parse_args()

    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    tasks = []
    for level in [int(x) for x in args.levels.split(",")]:
        dataset = construct_kernelbench_dataset(level)
        pids = dataset.get_problem_ids()
        if args.limit_per_level:
            pids = pids[:args.limit_per_level]
        for pid in pids:
            problem = dataset.get_problem_by_id(pid)
            tasks.append({
                "level": level, "problem_id": pid, "problem": problem.name,
                "ref_src": problem.code,
                "prompt": get_custom_prompt(
                    args.prompt_tier, ref_arch_src=problem.code,
                    backend="cutile", option="one_shot", precision="fp32"),
            })

    per_task = collections.defaultdict(lambda: {"n": 0, "passed": 0, "rewards": []})

    if args.from_run:
        by_pid = {t["problem_id"]: i for i, t in enumerate(tasks)}
        n_recs = 0
        for line in open(args.from_run):
            rec = json.loads(line)
            pid = int(rec["key"].split(":")[0])
            i = by_pid.get(pid)
            if i is None:
                continue
            r = reward_for(rec)
            if r is None:
                continue
            n_recs += 1
            acc = per_task[i]
            acc["n"] += 1
            acc["rewards"].append(r)
            acc["passed"] += 1 if rec["passed"] else 0
        print("derived from %d verified records in %s" % (n_recs, args.from_run))
    else:
        print("screening %d tasks x %d rollouts" % (len(tasks), args.samples))

        chat = Chat(args.base_url, args.model, args.temperature, args.top_p,
                    args.top_k, args.max_tokens)
        messages, index = [], []
        for i, t in enumerate(tasks):
            for _ in range(args.samples):
                messages.append([{"role": "user", "content": t["prompt"]}])
                index.append(i)

        texts = sample_batch(chat, messages, args.concurrency)

        items = []
        for j, (i, text) in enumerate(zip(index, texts)):
            code = extract_code(text) if not text.startswith("__ERROR__") else None
            if code:
                items.append(("%d" % j, code, tasks[i]["ref_src"]))

        with VerifierPool(workers=args.verify_workers, gpus=args.gpus) as pool:
            results = pool.verify_batch(items)

        for j, i in enumerate(index):
            rec = results.get("%d" % j)
            acc = per_task[i]
            acc["n"] += 1
            if rec is None:
                acc["rewards"].append(0.0)  # no code extracted
                continue
            r = reward_for(rec)
            if r is None:
                acc["n"] -= 1               # inconclusive; do not count it
                continue
            acc["rewards"].append(r)
            acc["passed"] += 1 if rec["passed"] else 0

    frontier, always_fail, always_pass = [], 0, 0
    for i, t in enumerate(tasks):
        acc = per_task[i]
        if acc["n"] == 0:
            continue
        rate = acc["passed"] / acc["n"]
        rewards = acc["rewards"]
        spread = max(rewards) - min(rewards) if rewards else 0.0
        entry = dict(t)
        entry.pop("prompt")                 # regenerated on use; keeps the file small
        entry.update(pass_rate=round(rate, 3), reward_spread=round(spread, 3),
                     n=acc["n"], category=category_of(t["ref_src"]))
        if args.mode == "solid":
            # The opposite selection: keep only what the policy already solves
            # every time. Correctness contributes no advantage on these -- it is
            # constant across the group -- which is exactly what makes them the
            # right material for learning speed, since the speed term then
            # supplies the entire gradient. They are what the default mode
            # discards, 326 of them in the last screen.
            if rate == 1:
                always_pass += 1
                frontier.append(entry)
            elif rate == 0:
                always_fail += 1
            continue

        if 0 < rate < 1:
            frontier.append(entry)
        elif rate == 0:
            always_fail += 1
            # A task nobody solves can still teach, if the failures differ: the
            # graded reward separates "will not build" from "ran but wrong".
            if spread > 0:
                entry["graded_only"] = True
                frontier.append(entry)
        else:
            always_pass += 1

    frontier.sort(key=lambda e: (-e["reward_spread"], -e["pass_rate"]))

    if args.category_quota:
        frontier = apply_category_quota(frontier, args.category_quota)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(frontier, f, indent=2)

    graded = sum(1 for e in frontier if e.get("graded_only"))
    solid = args.mode == "solid"
    print("\n%d tasks screened (mode: %s)" % (len(tasks), args.mode))
    print("  always fails      %d%s"
          % (always_fail, "  (dropped)" if solid else
             " (%d of them still usable via graded reward)" % graded))
    print("  always passes     %d  %s"
          % (always_pass, "(kept: correctness is constant, so the speed term "
                          "carries the gradient)" if solid
                          else "(no correctness gradient; dropped)"))
    print("  usable frontier   %d" % len(frontier))
    bycat = collections.Counter(e["level"] for e in frontier)
    print("  by level: %s" % dict(sorted(bycat.items())))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
