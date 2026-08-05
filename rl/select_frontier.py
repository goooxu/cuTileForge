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
sys.path.insert(0, HERE)

from repair_loop import Chat, extract_code, sample_batch  # noqa: E402
from reward import reward_for  # noqa: E402
from worker import VerifierPool  # noqa: E402


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
                    "cutile_docs", ref_arch_src=problem.code, backend="cutile",
                    option="one_shot", precision="fp32"),
            })

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

    per_task = collections.defaultdict(lambda: {"n": 0, "passed": 0, "rewards": []})
    for j, i in enumerate(index):
        rec = results.get("%d" % j)
        acc = per_task[i]
        acc["n"] += 1
        if rec is None:
            acc["rewards"].append(0.0)      # no code extracted
            continue
        r = reward_for(rec)
        if r is None:
            acc["n"] -= 1                   # inconclusive; do not count it
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
                     n=acc["n"])
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(frontier, f, indent=2)

    graded = sum(1 for e in frontier if e.get("graded_only"))
    print("\n%d tasks screened" % len(tasks))
    print("  always fails      %d (%d of them still usable via graded reward)"
          % (always_fail, graded))
    print("  always passes     %d  (no gradient; dropped)" % always_pass)
    print("  usable frontier   %d" % len(frontier))
    bycat = collections.Counter(e["level"] for e in frontier)
    print("  by level: %s" % dict(sorted(bycat.items())))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
