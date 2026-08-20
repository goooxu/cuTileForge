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

HELDOUT_LEVELS = frozenset({60, 84, 88, 97, 98, 99})


def in_speed_band(best, min_speedup, max_speedup) -> bool:
    """Keep tasks whose best passing sample sits strictly inside (min, max).

    None on either bound means unbounded. A missing timing is never in the band:
    selecting on speed without measurements is how the last speed run trained on
    tasks it already won by 6x.
    """
    if best is None:
        return False
    if max_speedup is not None and best >= max_speedup:
        return False
    if min_speedup is not None and best <= min_speedup:
        return False
    return True


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
    ap.add_argument("--max-tokens", type=int, default=32768)
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
    ap.add_argument("--max-speedup", type=float, default=None,
                    help="With --mode solid, keep only tasks whose best sample "
                         "is slower than this multiple of the reference. Use 1.0 "
                         "to train on the regime the benchmark occupies. Requires "
                         "the harvest to have been verified with --measure-time.")
    ap.add_argument("--min-speedup", type=float, default=None,
                    help="With --mode solid, drop tasks whose best sample is at "
                         "or below this multiple. The speed bonus clamps at "
                         "0.25x, so training below that rebuilds the dead zone "
                         "on the other side. Requires --measure-time harvest.")
    ap.add_argument("--category-quota", default=None,
                    help="Comma-separated cat=N caps, '*=N' for the rest. "
                         "Without this the frontier's composition is whatever "
                         "the task pool happened to contain.")
    ap.add_argument("--from-run", default=None,
                    help="Skip sampling and derive the frontier from an existing "
                         "verified run (JSONL from verify/fast_verify.py). A "
                         "k=16 harvest already measures each task's pass rate "
                         "far better than a k=6 screen would.")
    ap.add_argument("--rollouts-out", default=None,
                    help="Write sampled texts here (JSONL) before verifying.")
    ap.add_argument("--from-rollouts", default=None,
                    help="Skip sampling; verify texts previously written by "
                         "--rollouts-out. The verifier cannot share a GPU with "
                         "a second docker that is already serving vLLM.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Sample only. Pair with --rollouts-out, then stop vLLM "
                         "and re-run with --from-rollouts.")
    args = ap.parse_args()
    if args.no_verify and not args.rollouts_out:
        raise SystemExit("--no-verify needs --rollouts-out")
    if args.from_rollouts and args.no_verify:
        raise SystemExit("--from-rollouts cannot be combined with --no-verify")
    n_src = sum(bool(x) for x in (args.from_run, args.from_rollouts))
    if n_src > 1:
        raise SystemExit("pick at most one of --from-run / --from-rollouts")

    levels = [int(x) for x in args.levels.split(",")]
    leaked = [lv for lv in levels if lv in HELDOUT_LEVELS]
    if leaked:
        raise SystemExit("refusing held-out levels %s" % leaked)

    if ((args.max_speedup is not None or args.min_speedup is not None)
            and not args.from_run and not args.from_rollouts):
        raise SystemExit(
            "--min-speedup/--max-speedup need --from-run of a harvest verified "
            "with --measure-time. Live screening does not time candidates.")

    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    tasks = []
    for level in levels:
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
            if rec.get("speedup"):
                acc["best_speedup"] = max(acc.get("best_speedup") or 0.0,
                                          rec["speedup"])
        print("derived from %d verified records in %s" % (n_recs, args.from_run))
        timed = sum(1 for a in per_task.values() if a.get("best_speedup"))
        print("  %d of %d tasks carry a timing" % (timed, len(per_task)))
        if ((args.max_speedup is not None or args.min_speedup is not None)
                and timed == 0):
            raise SystemExit(
                "--min-speedup/--max-speedup was given but no record carries a "
                "speedup. The harvest was verified without --measure-time, so "
                "selecting on speed here would silently select on nothing -- "
                "which is exactly how the last speed run ended up training on "
                "tasks it already won by 6x.")
    else:
        print("screening %d tasks x %d rollouts" % (len(tasks), args.samples))

        if args.from_rollouts:
            index, texts = [], []
            for line in open(args.from_rollouts):
                rec = json.loads(line)
                index.append(int(rec["i"]))
                texts.append(rec["text"])
            print("loaded %d rollouts from %s" % (len(texts), args.from_rollouts))
        else:
            chat = Chat(args.base_url, args.model, args.temperature, args.top_p,
                        args.top_k, args.max_tokens)
            messages, index = [], []
            for i, t in enumerate(tasks):
                for _ in range(args.samples):
                    messages.append([{"role": "user", "content": t["prompt"]}])
                    index.append(i)

            texts = sample_batch(chat, messages, args.concurrency)
            if args.rollouts_out:
                os.makedirs(os.path.dirname(args.rollouts_out) or ".", exist_ok=True)
                tmp = args.rollouts_out + ".tmp"
                with open(tmp, "w") as f:
                    for i, text in zip(index, texts):
                        f.write(json.dumps({"i": i, "text": text}) + "\n")
                os.replace(tmp, args.rollouts_out)
                print("wrote %d rollouts to %s" % (len(texts), args.rollouts_out))

        n_err = sum(1 for t in texts if t.startswith("__ERROR__"))
        n_code = 0
        items = []
        for j, (i, text) in enumerate(zip(index, texts)):
            code = extract_code(text) if not text.startswith("__ERROR__") else None
            if code:
                n_code += 1
                items.append(("%d" % j, code, tasks[i]["ref_src"]))
        print("extracted code from %d/%d rollouts (%d transport errors)"
              % (n_code, len(texts), n_err))

        if args.no_verify:
            print("skipping verify (--no-verify)")
            return

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
        if args.mode == "solid" and (args.max_speedup is not None
                                     or args.min_speedup is not None):
            # Keep only what the policy solves reliably *and* sits in the
            # speed band the benchmark occupies. Without the upper bound the
            # last speed run trained on tasks it already won by 6x to 11x.
            # Without the lower bound it would train inside the 0.25x clamp,
            # where the bonus is flat again. Per-problem speed on the
            # benchmark did not move at all: median ratio 1.000x.
            best = acc.get("best_speedup")
            if not in_speed_band(best, args.min_speedup, args.max_speedup):
                continue
            entry["best_speedup"] = round(best, 4)

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
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(frontier, f, indent=2)
    os.replace(tmp, args.out)

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
    speeds = [e["best_speedup"] for e in frontier if "best_speedup" in e]
    if speeds:
        speeds.sort()
        print("  best_speedup      min %.3f  median %.3f  max %.3f  (n=%d)"
              % (speeds[0], speeds[len(speeds) // 2], speeds[-1], len(speeds)))
    bycat = collections.Counter(e["level"] for e in frontier)
    print("  by level: %s" % dict(sorted(bycat.items())))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
