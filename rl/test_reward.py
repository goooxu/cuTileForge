"""Check the reward's shape against real verifier output.

Two things have to hold or GRPO will optimise the wrong thing: correctness must
outrank any speed, and the grading must actually produce spread on the failures,
since a group of eight failures with identical rewards contributes no gradient.

Runs against a verified.jsonl from a real run rather than synthetic records, so
it also catches stages the verifier emits that the reward forgot to handle.

Usage:
    python3 rl/test_reward.py --verified runs/repair_l94_verified.jsonl
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import STAGE_REWARD, reward_for, speed_bonus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default=None,
                    help="A fast_verify.py output to score. Optional.")
    args = ap.parse_args()

    fails = []

    def check(cond, msg):
        print("  %-4s %s" % ("ok" if cond else "FAIL", msg))
        if not cond:
            fails.append(msg)

    print("ordering:")
    slowest_correct = reward_for({"passed": True, "stage": "pass", "speedup": 0.01})
    fastest_wrong = reward_for({"passed": False, "stage": "exec",
                                "error": "output mismatch, max diff 3",
                                "speedup": 100.0})
    check(slowest_correct > fastest_wrong,
          "a correct but very slow kernel (%.2f) outranks any incorrect one (%.2f)"
          % (slowest_correct, fastest_wrong))

    fastest_correct = reward_for({"passed": True, "stage": "pass", "speedup": 64.0})
    check(fastest_correct - slowest_correct <= 0.31,
          "speed moves reward by at most 0.3 (%.2f to %.2f)"
          % (slowest_correct, fastest_correct))
    check(speed_bonus(4.0) == speed_bonus(64.0),
          "the speed bonus saturates rather than rewarding runaway numbers")
    check(speed_bonus(1.0) == 0.0 and speed_bonus(0.5) == 0.0,
          "no bonus for merely matching or losing to torch")

    print("\ngrading of failures:")
    wont_build = reward_for({"passed": False, "stage": "exec",
                             "error": "TileTypeError: Grid dimensions must be"})
    ran_wrong = reward_for({"passed": False, "stage": "exec",
                            "error": "AssertionError: output mismatch, max diff 1.7"})
    check(ran_wrong > wont_build,
          "a kernel that ran and got wrong numbers (%.1f) beats one that will not "
          "build (%.1f)" % (ran_wrong, wont_build))
    check(reward_for({"passed": False, "stage": "purity",
                      "error": "uses torch.matmul"}) == 0.0,
          "falling back to PyTorch earns nothing")

    print("\nharness failures:")
    check(reward_for({"passed": False, "stage": "oom"}) is None,
          "OOM is inconclusive, not a zero")
    check(reward_for({"passed": False, "stage": "worker_crash"}) is None,
          "a dead verifier worker is inconclusive, not a zero")
    check(reward_for({"passed": False, "stage": "cuda_poison"}) is None,
          "a sticky CUDA context is inconclusive, not a zero")

    if args.verified:
        print("\nagainst %s:" % os.path.basename(args.verified))
        recs = [json.loads(l) for l in open(args.verified)]
        seen_stages = collections.Counter(r.get("stage", "") for r in recs)
        unknown = [s for s in seen_stages
                   if s not in STAGE_REWARD
                   and s not in ("oom", "worker_crash", "cuda_poison")]
        check(not unknown, "every stage the verifier emits is handled: %s"
              % (", ".join(sorted(seen_stages)) or "none"))

        rewards = [reward_for(r) for r in recs]
        rewards = [r for r in rewards if r is not None]
        dist = collections.Counter(round(r, 2) for r in rewards)
        print("  reward distribution over %d records:" % len(rewards))
        for val, n in sorted(dist.items()):
            print("    %.2f  %5d  %s" % (val, n, "#" * (n * 40 // len(rewards))))
        check(len(dist) > 1, "the reward separates the run into more than one value")

    print("\n%s" % ("all checks passed" if not fails else
                    "%d FAILED" % len(fails)))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
