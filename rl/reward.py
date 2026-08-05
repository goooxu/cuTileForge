"""Graded reward for GRPO over generated cuTile kernels.

A binary correct/incorrect reward would leave most groups flat. At pass@1 near
14%, a group of eight samples is usually eight failures, every advantage is zero
and the step contributes no gradient at all. Grading the failures is what gives
those groups something to learn from, so it is a requirement here rather than a
refinement.

The ordering is deliberate. Correctness always outranks any amount of speed, so
the fourth round's mistake -- a model that got more answers right while getting
slower -- cannot be rewarded. Speed is a bonus on top of a correct kernel and is
capped well below the step from wrong to right.

Reward comes from the same verifier the benchmark uses, so what is rewarded here
and what is reported there cannot drift apart.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "verify"))

from worker import verify_and_time  # noqa: E402

# Partial credit by how far a candidate got. The gaps are even so that no single
# stage transition dominates, and every one of them is well under the 0.4 step
# from "runs but wrong" to "correct".
STAGE_REWARD = {
    "purity": 0.0,      # wrote torch, or no real cuTile kernel
    "no_code": 0.0,     # no code block at all
    "exec": 0.2,        # pure cuTile, but fails to import, compile or launch
    "timeout": 0.2,     # a kernel that never finishes is no better than one that
                        # does not build
    "pass": 1.0,
}

# The verifier files both "will not compile" and "ran and produced wrong numbers"
# under stage "exec", but they are not equally close to working: the second one
# built, launched and returned an array of the right shape. Convolution failures
# in particular sit almost entirely in the second group, so collapsing them would
# throw away the gradient that matters most there.
RAN_BUT_WRONG = 0.6
_RAN_BUT_WRONG_SIGNS = ("output mismatch", "non-finite", "!= expected")

# Reward for a correct kernel that is also fast. Capped at +0.3 so that no
# speedup can make an incorrect kernel look competitive with a correct one, and
# saturating at 4x, beyond which further gains are not worth chasing.
SPEED_BONUS_MAX = 0.3
SPEED_SATURATES_AT = 4.0


def speed_bonus(speedup) -> float:
    if not speedup or speedup <= 1.0:
        return 0.0
    frac = math.log2(speedup) / math.log2(SPEED_SATURATES_AT)
    return SPEED_BONUS_MAX * min(frac, 1.0)


def reward_for(rec: dict) -> float:
    """Map one verifier record to a scalar reward."""
    stage = rec.get("stage", "")
    if stage in ("oom", "worker_crash"):
        # The harness failed, not the candidate. Signalling zero here would
        # punish a kernel for the verifier's own contention.
        return None
    if rec.get("passed"):
        return STAGE_REWARD["pass"] + speed_bonus(rec.get("speedup"))
    if stage == "exec":
        err = rec.get("error") or ""
        if any(s in err for s in _RAN_BUT_WRONG_SIGNS):
            return RAN_BUT_WRONG
    return STAGE_REWARD.get(stage, 0.2)


def score_rollouts(items, gpus: int = 4, workers: int = 16,
                   measure_speed: bool = True, progress=None) -> dict:
    """Verify (key, code, ref_src) triples and return {key: (reward, record)}.

    Candidates whose code could not be extracted should be passed with code=None
    and are scored zero without reaching the verifier.
    """
    verifiable = [(k, c, r) for k, c, r in items if c]
    out = {k: (0.0, {"stage": "no_code", "passed": False})
           for k, c, _ in items if not c}

    if verifiable:
        if measure_speed:
            recs = verify_and_time(verifiable, workers=workers, gpus=gpus,
                                   progress=progress)
        else:
            from worker import VerifierPool
            with VerifierPool(workers=workers, gpus=gpus) as pool:
                recs = pool.verify_batch(verifiable)
        for key, rec in recs.items():
            out[key] = (reward_for(rec), rec)
    return out


def summarise(scored: dict) -> dict:
    """Aggregate counts a training loop should watch every iteration."""
    rewards = [r for r, _ in scored.values() if r is not None]
    recs = [rec for r, rec in scored.values() if r is not None]
    n = max(len(rewards), 1)
    speeds = [rec.get("speedup") for rec in recs if rec.get("speedup")]
    return {
        "n": len(rewards),
        "mean_reward": sum(rewards) / n,
        "pass_rate": sum(1 for rec in recs if rec.get("passed")) / n,
        # Purity is the canary for reward hacking: the cheapest way to raise a
        # correctness reward is to quietly hand the work back to PyTorch.
        "purity_rate": 1.0 - sum(1 for rec in recs
                                 if rec.get("stage") == "purity") / n,
        "no_code_rate": sum(1 for rec in recs
                            if rec.get("stage") == "no_code") / n,
        "fast_rate": sum(1 for s in speeds if s > 1.0) / n,
        "inconclusive": sum(1 for r, _ in scored.values() if r is None),
    }
