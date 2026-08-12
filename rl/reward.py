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

# Partial credit for a purity failure that is nearly a port. This is the largest
# group of unsolved benchmark problems -- a real cuTile kernel doing the hard work
# with one pointwise activation left as torch.relu(...) -- and a flat 0.0 scores it
# the same as an empty response, so nothing tells the model it was one edit away.
#
# The anti-hacking guarantee is kept by construction: this band tops out strictly
# below STAGE_REWARD["exec"], so *any* genuinely pure attempt scores at least as
# much as *any* impure one, and handing work back to PyTorch can never pay. Two
# hard zeros remain -- no real launched kernel at all, and delegating to a
# torch.nn compute layer -- since both are wholesale delegation rather than an
# unfinished port.
PURITY_NEARLY_MAX = 0.18
PURITY_NEARLY_MIN = 0.04
PURITY_HOPELESS_OPS = 8          # this many leftovers is not an unfinished port

# A near-port that also computes the right answer is closer to passing than one
# that does not, so it gets its own band -- but that band stays under
# STAGE_REWARD["exec"], and the reason is measured rather than assumed.
#
# It was tried at 0.20-0.30, which puts an impure-but-correct kernel above pure
# cuTile that fails to compile. That broke the invariant the lower band exists to
# preserve, and it backfired on its own target: pass@1 rose 51.6% to 55.8% while
# problems solved fell 150 to 147, and Level 2's purity failures -- the whole
# point of the change -- grew from 25 of 35 unsolved to 31 of 38. Of the 11
# Level 2 problems that run lost, 9 went from a fully pure passing kernel to a
# correct one with torch left in. The policy took the near-port instead of
# finishing it, exactly the local optimum this comment used to warn about.
#
# Training metrics could not see it: purity_rate held at 0.971 and pass_rate kept
# climbing, because the synthetic chains are short enough that purity is already
# at its ceiling there and the local optimum is no temptation. Only the benchmark
# showed it. The lesson is that the invariant was doing real work, not costing
# resolution -- the earlier round's shortfall on Level 2 was not a reward-scale
# problem.
PURITY_CORRECT_MAX = 0.19
PURITY_CORRECT_MIN = 0.10


def purity_credit(rec: dict) -> float:
    """Grade a purity failure by how much torch is left, and whether it works."""
    if not rec.get("has_real_kernel") or rec.get("delegates_to_nn"):
        return 0.0
    n = rec.get("torch_ops_left")
    if not n:
        # The gate failed for some other reason (a missing launch, no tile ops);
        # not an unfinished port, so no credit.
        return 0.0
    frac = min(max((n - 1) / float(PURITY_HOPELESS_OPS - 1), 0.0), 1.0)
    if rec.get("impure_correct"):
        hi, lo = PURITY_CORRECT_MAX, PURITY_CORRECT_MIN
    else:
        hi, lo = PURITY_NEARLY_MAX, PURITY_NEARLY_MIN
    return hi - frac * (hi - lo)

# The verifier files both "will not compile" and "ran and produced wrong numbers"
# under stage "exec", but they are not equally close to working: the second one
# built, launched and returned an array of the right shape. Convolution failures
# in particular sit almost entirely in the second group, so collapsing them would
# throw away the gradient that matters most there.
RAN_BUT_WRONG = 0.6
_RAN_BUT_WRONG_SIGNS = ("output mismatch", "non-finite", "!= expected")

# Within that group, grade by how wrong. This band is not a minor case: 61 of the
# 66 benchmark problems the best model cannot solve land in it, so a flat 0.6 for
# everything from "off in the last digit" to "off by six orders of magnitude"
# leaves no gradient exactly where all the remaining headroom is. It also matters
# for group-relative advantage, which is zero for a group whose members all score
# the same.
#
# Interpolates between RAN_BUT_WRONG at the tolerance and RAN_BUT_WRONG_FLOOR at
# RAN_BUT_WRONG_HOPELESS, on a log scale because relative error spans orders of
# magnitude. Never reaches 1.0: only the verifier's own allclose decides that.
RAN_BUT_WRONG_FLOOR = 0.3
RAN_BUT_WRONG_TOL = 1e-4          # matches the verifier's atol/rtol
RAN_BUT_WRONG_HOPELESS = 1.0      # relative error of 100% or more


def numeric_credit(rel_diff) -> float:
    """Partial credit for a kernel that ran but got the numbers wrong.

    Falls back to the flat value when the verifier did not record a deviation,
    so older runs and other failure shapes keep their previous score.
    """
    if rel_diff is None:
        return RAN_BUT_WRONG
    if rel_diff <= RAN_BUT_WRONG_TOL:
        # Just outside allclose on one element; nearly right.
        return RAN_BUT_WRONG
    span = math.log10(RAN_BUT_WRONG_HOPELESS / RAN_BUT_WRONG_TOL)
    frac = math.log10(rel_diff / RAN_BUT_WRONG_TOL) / span
    frac = min(max(frac, 0.0), 1.0)
    return RAN_BUT_WRONG - frac * (RAN_BUT_WRONG - RAN_BUT_WRONG_FLOOR)

# Reward for a correct kernel that is also fast. Capped at +0.3 so that no
# speedup can make an incorrect kernel look competitive with a correct one, and
# saturating at 4x, beyond which further gains are not worth chasing.
#
# The speedup this reads is measured against torch.compile, not eager -- see
# verify/worker.py. Trained against eager the bonus rewards beating intermediate
# materialisation, which inductor removes for free, so it was paying for wins a
# user would never see: the run that looked best on eager (60 problems faster)
# was worse than its predecessor once re-measured against inductor (41 vs 45).
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
    if stage == "purity":
        return purity_credit(rec)
    if stage == "exec":
        err = rec.get("error") or ""
        if any(s in err for s in _RAN_BUT_WRONG_SIGNS):
            return numeric_credit(rec.get("rel_diff"))
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
        # How much of the batch is in the graded band, and how close it is. This
        # is the band the remaining headroom lives in, so it is worth watching
        # directly rather than inferring from mean reward.
        "wrong_numbers_rate": sum(1 for rec in recs
                                  if rec.get("rel_diff") is not None) / n,
        # Impure but correct: the band that now pays 0.3. If this climbs while
        # pass_rate does not, the policy has settled for the near-port instead of
        # finishing it, which is the failure mode that band risks.
        "impure_correct_rate": sum(1 for rec in recs
                                   if rec.get("impure_correct")) / n,
        "median_rel_diff": _median([rec["rel_diff"] for rec in recs
                                    if rec.get("rel_diff") is not None]),
    }


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None
