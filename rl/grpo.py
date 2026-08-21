"""GRPO over cuTile kernel generation.

The case for reinforcement learning here is narrow and specific. The model solves
25% of the benchmark at pass@4 but only 14.5% at pass@1: it already knows how to
write a correct kernel for a quarter of the problems and simply does not do so
reliably. Concentrating probability mass on what already works is what policy
gradient is for, and closing even half that gap would beat every supervised round
so far. It will do nothing for the 102 problems no version has ever solved; those
need data, not a better objective.

Group-relative advantage, so no value network -- a critic at this scale would cost
more than the policy. Rollouts come from vLLM because generating with transformers
across a sharded 80B model is far slower than the training step it feeds, which
makes the loop off-policy between refreshes; PPO-style ratio clipping is what
makes that sound, and the clip fraction is reported every iteration so the
staleness is visible rather than assumed.

A KL penalty against the starting policy is not optional here, and it is free.
Running without one collapsed the policy outright: the share of rollouts with no
extractable code went from 1.6% at iteration 22 to 99.2% by iteration 28, and the
model simply stopped writing code. The mechanism is not exotic. Most rollouts
fail, so most group-relative advantages are negative, and the cheapest way to
lower the probability of every failing completion at once is to stop emitting
code at all. Nothing in a clipped objective bounds that when --inner-epochs is 1,
because the ratio is then identically 1 and the clip never binds.

It is free because of --fresh-lora. An earlier version of this note argued a
reference policy would either undo the supervised rounds or cost another 27.5 GB
of weights. Both are wrong once the adapter sits on top of already-merged
supervised weights: a zero LoRA contribution *is* the supervised policy, so the
reference log-probabilities cost one extra forward rather than a second copy of
the weights, and anchoring to it pulls toward the SFT model rather than away
from it. peft's disable_adapter() is the wrong way to get that zero: it skips
the LoRA add entirely, which on this Glimmer stack is a different kernel path
from B=0, and the two disagree by ~0.3 nats/token on 8k sequences -- enough for
the k3 KL estimator to read thousands. Zeroing the LoRA scaling keeps the same
path the policy uses.

Only the attention and DeltaNet adapters are trained. The first run of this
script did that by loading an adapter trained with the full target set and
freezing the experts, which left 34M trainable -- 0.04% of the model -- and
moved nothing in 20 iterations. Prefer --fresh-lora against an already-merged
policy instead: a new rank-128 adapter over merged weights starts at exactly the
same policy (B is zero-initialised) but has 137M trainable, and still checkpoints
in 537 MB. See train/lora_config.py.

Usage:
    python3 rl/grpo.py --model /raid/tmp/.../model-F --fresh-lora \\
        --prompt-tier cutile_concepts --frontier runs/rl_frontier.json \\
        --out runs/grpo --iterations 20
"""

import argparse
import atexit
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "repair"))
sys.path.insert(0, os.path.join(HERE, "..", "verify"))
sys.path.insert(0, HERE)

from lora_config import (family_of, freeze_experts,  # noqa: E402
                         load_base_model, logit_transform, targets_for,
                         validate_targets)
from repair_loop import Chat, extract_code, sample_batch  # noqa: E402
from reward import score_rollouts, summarise  # noqa: E402
from rollout_tokens import build_sequence  # noqa: E402
from train_lora import TURN_END, unwrap  # noqa: E402
from worker import VerifierPool  # noqa: E402


# Eval suite and the two held-out tracks. Training on these would leak the
# published numbers; GRPO's frontier has to come from the training levels.
HELDOUT_LEVELS = frozenset({60, 84, 88, 97, 98, 99})


def completion_logprobs(model, ids, comp_mask, grad: bool):
    """Per-token log probability of the completion tokens.

    Applies the LM head only at the positions being scored. The prompt is ~14k
    tokens of documentation and the completion is a fraction of that, so
    projecting everything through a 152k vocabulary would cost gigabytes to
    produce numbers that are immediately discarded.

    Calling the head directly also skips whatever the model's forward does after
    it, which on some architectures is a logit softcap; without it these would
    be log probabilities of a distribution the model does not have.
    """
    body, lm_head = unwrap(model)
    transform = logit_transform(body)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        hidden = body(input_ids=ids,
                      attention_mask=torch.ones_like(ids)).last_hidden_state
        # Position t predicts token t+1, so a completion token at t is scored by
        # the hidden state at t-1.
        h = hidden[:, :-1]
        target = ids[:, 1:].to(h.device)
        keep = comp_mask[:, 1:].to(h.device).bool()
        if not keep.any():
            return None
        logits = lm_head(h[keep])
        if transform is not None:
            logits = transform(logits)
        return torch.log_softmax(logits.float(), dim=-1).gather(
            1, target[keep].unsqueeze(1)).squeeze(1)


def group_advantages(rewards):
    """Normalise within a group; a group with no spread contributes nothing."""
    vals = [r for r in rewards if r is not None]
    if len(vals) < 2:
        return [0.0] * len(rewards)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    if std < 1e-6:
        return [0.0] * len(rewards)
    return [0.0 if r is None else (r - mean) / std for r in rewards]


@contextmanager
def adapter_kept_at_base(model):
    """Reference forwards through the LoRA compute path with contribution 0.

    disable_adapter() takes the `if self.disable_adapters: return F.linear(x, W)`
    branch. A fresh LoRA takes `F.linear(x, W) + B(A(x)) * scaling` with B=0.
    Those are not the same kernels, and Glimmer's residual stream accumulates
    the difference over long sequences. Setting scaling to 0 keeps the add.
    """
    restored = []
    found = 0
    for m in model.modules():
        if not (hasattr(m, "lora_A") or hasattr(m, "lora_embedding_A")):
            continue
        found += 1
        scaling = getattr(m, "scaling", None)
        if isinstance(scaling, dict):
            prev = dict(scaling)

            def _restore_dict(sc=scaling, p=prev):
                sc.clear()
                sc.update(p)

            restored.append(_restore_dict)
            for k in list(scaling):
                v = scaling[k]
                scaling[k] = v * 0 if hasattr(v, "__mul__") else 0.0
            continue
        if scaling is not None:
            def _restore_attr(mod=m, p=scaling):
                mod.scaling = p

            restored.append(_restore_attr)
            m.scaling = scaling * 0 if hasattr(scaling, "__mul__") else 0.0
            continue
        for name, p in m.named_parameters(recurse=False):
            if "lora_B" not in name and "lora_embedding_B" not in name:
                continue
            data = p.data.detach().clone()

            def _restore_B(param=p, d=data):
                param.data.copy_(d)

            restored.append(_restore_B)
            p.data.zero_()
    if found == 0:
        raise RuntimeError("adapter_kept_at_base found no LoRA layers")
    try:
        yield
    finally:
        for fn in restored:
            fn()


def adapter_logprob_l1(model, dev, length=16):
    """Mean |logp_on - logp_base|. A fresh LoRA is B=0, so this must be ~0.

    The 16-token probe missed the GL-C failure: iter 0 reported kl 13199
    (loss almost equal to kl_coef * KL) on real 20k-token sequences. Probe
    at `length` close to a training example, not just a toy tensor.
    """
    ids = torch.arange(8, 8 + length, device=dev, dtype=torch.long).unsqueeze(0)
    mask = torch.ones_like(ids)
    mask[:, : min(4, length // 4)] = 0
    with torch.no_grad():
        on = completion_logprobs(model, ids, mask, grad=False)
        with adapter_kept_at_base(model):
            off = completion_logprobs(model, ids, mask, grad=False)
    if on is None or off is None:
        return None
    return float((on - off).abs().mean())


def _set_checkpointing(model, enabled: bool) -> None:
    """Reentrant checkpointing on Glimmer/Gated-DeltaNet corrupts the backward.

    The on/off logprob divergence on long sequences was disable_adapter taking
    a different kernel path, not this flag; keep non-reentrant anyway.
    """
    if enabled:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base weights.")
    ap.add_argument("--adapter", default=None,
                    help="Starting policy adapter. Omit with --fresh-lora.")
    ap.add_argument("--fresh-lora", action="store_true",
                    help="Attach a new attention-only rank-128 adapter instead "
                         "of loading one. Use when --model is already merged: "
                         "the policy starts unchanged and every trainable "
                         "parameter is one RL can actually move.")
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--prompt-tier", default="cutile_docs",
                    help="Prompt composition for rollouts. Must match what the "
                         "policy was trained on, and must match the tier the "
                         "frontier was screened at.")
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--prompts-per-iter", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--inner-epochs", type=int, default=1,
                    help="Gradient passes per rollout. At 1 the ratio is exactly "
                         "1 and clipping never binds, so this degenerates to "
                         "REINFORCE with a group baseline -- correct, but the "
                         "old-logprob pass is then wasted work. Above 1 the "
                         "clipping starts doing its job.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=20480,
                    help="Drop rollouts longer than this (prompt + sampled "
                         "text). 20480 matches Glimmer SFT; front-truncation "
                         "would strip the task description.")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Needed at max-len 20480; off by default because it "
                         "makes each step slower.")
    ap.add_argument("--kl-coef", type=float, default=0.05,
                    help="Weight on KL to the SFT policy (LoRA scaling held at "
                         "0). Set to 0 only if you want to reproduce the collapse.")
    ap.add_argument("--max-no-code", type=float, default=0.35,
                    help="Abort if this share of rollouts yields no code block. "
                         "The failure mode this catches is degeneration to empty "
                         "output, which is silent in the reward until it is total.")
    ap.add_argument("--snapshot-every", type=int, default=5,
                    help="Keep a numbered copy of the adapter this often. The "
                         "rolling checkpoint alone is not enough: a collapse "
                         "overwrites the good weights with the broken ones.")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--served-model", default="Qwen3-Coder-Next")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--verify-workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--no-speed", action="store_true",
                    help="Skip timing; correctness-only reward.")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint directory from an earlier invocation.")
    ap.add_argument("--reasoning-strength", default=None,
                    help="Passed to the chat template so the training prefix "
                         "matches serving (Muse Glimmer writes it into the "
                         "system block).")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    workers = args.verify_workers
    if not args.no_speed and workers > args.gpus:
        print("timing needs exclusive GPUs; clamping verify-workers %d -> %d"
              % (workers, args.gpus))
        workers = args.gpus

    from transformers import AutoConfig, AutoTokenizer
    from peft import PeftModel
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    frontier = json.load(open(args.frontier))
    leaked = sorted({t["level"] for t in frontier if t["level"] in HELDOUT_LEVELS})
    if leaked:
        raise SystemExit("frontier contains held-out levels %s" % leaked)
    print("frontier: %d tasks" % len(frontier))

    # Prompts are rebuilt rather than stored, so they stay identical to what
    # evaluation composes.
    datasets, prompts, refs = {}, {}, {}
    for t in frontier:
        lvl = t["level"]
        if lvl not in datasets:
            datasets[lvl] = construct_kernelbench_dataset(lvl)
        problem = datasets[lvl].get_problem_by_id(t["problem_id"])
        key = (lvl, t["problem_id"])
        refs[key] = problem.code
        prompts[key] = get_custom_prompt(
            args.prompt_tier, ref_arch_src=problem.code, backend="cutile",
            option="one_shot", precision="fp32")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    family = family_of(AutoConfig.from_pretrained(args.model,
                                                  trust_remote_code=True))
    chat_kwargs = ({"reasoning_strength": args.reasoning_strength}
                   if args.reasoning_strength else {})
    turn_end = TURN_END.get(family)

    print("loading policy ...")
    t0 = time.time()
    # 30B bf16 is ~60 GB and fits on one 184 GB card, so device_map="auto"
    # packs the whole policy onto a single GPU. Backward at max_len=20480
    # then OOMs at ~180 GB on that card. Cap per-GPU weight placement so
    # accelerate has to split.
    max_memory = None
    n_cuda = torch.cuda.device_count()
    if n_cuda > 1:
        per_gb = max(20, 60 // n_cuda + 8)
        max_memory = {i: "%dGiB" % per_gb for i in range(n_cuda)}
        print("forcing %d-way split (max_memory %s)" % (n_cuda, max_memory))
    model = load_base_model(args.model, device_map="auto", dtype=torch.bfloat16,
                            max_memory=max_memory)
    if args.fresh_lora:
        from peft import LoraConfig, get_peft_model
        targets = targets_for(model, "attention_only")
        validate_targets(model, targets)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=targets))
    else:
        if not args.adapter:
            raise SystemExit("pass --adapter or --fresh-lora")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    print("loaded in %.0fs" % (time.time() - t0))
    placed = {}
    for p in model.parameters():
        d = str(p.device)
        placed[d] = placed.get(d, 0) + p.numel()
    print("parameter devices: %s" %
          {k: "{:,}".format(v) for k, v in placed.items()})

    # A fresh attention-only adapter has no expert tensors to begin with, so this
    # is a no-op there and only bites when resuming one of the old full-target
    # adapters.
    trainable, frozen = freeze_experts(model)
    print("training %s of %s parameters (%.3f%%)"
          % ("{:,}".format(trainable), "{:,}".format(trainable + frozen),
             trainable / (trainable + frozen) * 100))
    if trainable == 0:
        raise SystemExit("nothing left trainable after freezing experts")

    if args.gradient_checkpointing:
        _set_checkpointing(model, True)
    model.enable_input_require_grads()
    # Dropout in the logprob path would make the KL estimator compare two
    # different stochastic forwards, not adapter on vs off. LoRA dropout is
    # already 0; eval() also freezes any base-model dropout.
    model.eval()
    model.config.use_cache = False

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), foreach=False)
    dev = next(model.parameters()).device

    # Spawn workers before any parent forward. A completion_logprobs probe
    # first was enough to make the next CUDA process see 0 devices, and
    # every iteration then skipped.
    print("verifier pool: %d workers on %d gpus" % (workers, args.gpus))
    pool = VerifierPool(workers=workers, gpus=args.gpus)
    atexit.register(pool.close)

    if args.fresh_lora and args.kl_coef > 0:
        l1 = None
        for probe_len in (16, 256):
            try:
                l1 = adapter_logprob_l1(model, dev, length=probe_len)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print("fresh LoRA vs scaling-0 probe OOM at len=%d (%s)"
                      % (probe_len, e))
                continue
            except Exception as e:
                print("fresh LoRA vs scaling-0 probe skipped at len=%d: %s"
                      % (probe_len, e))
                continue
            print("fresh LoRA vs scaling-0 logprob L1 (len %d): %s" %
                  (probe_len, "n/a" if l1 is None else "%.6f" % l1))
            if l1 is not None and l1 > 0.05:
                raise SystemExit(
                    "fresh LoRA is not an identity at len=%d (L1 %.4f). "
                    "The KL term would dominate the loss; refusing to train."
                    % (probe_len, l1))
        if l1 is None:
            print("fresh LoRA identity probe did not run; watch iter-0 KL")

    start_iter = 0
    if args.resume:
        state = torch.load(os.path.join(args.resume, "opt.pt"), map_location="cpu")
        opt.load_state_dict(state["optimizer"])
        start_iter = state["iteration"] + 1
        deltas = torch.load(os.path.join(args.resume, "trainable.pt"),
                            map_location="cpu")
        missing = model.load_state_dict(deltas, strict=False)
        print("resumed at iteration %d (%d tensors restored, %d unexpected)"
              % (start_iter, len(deltas), len(missing.unexpected_keys)))

    chat = Chat(args.base_url, args.served_model, args.temperature, args.top_p,
                args.top_k, args.max_tokens)

    # Carry the log forward across invocations, or the trend window restarts at
    # every resume and hides exactly the comparison it exists to show.
    history = []
    hist_path = os.path.join(args.out, "history.jsonl")
    if os.path.exists(hist_path):
        history = [json.loads(l) for l in open(hist_path)]

    for it in range(start_iter, start_iter + args.iterations):
        t_iter = time.time()
        # Seed per iteration rather than once per process. Seeding once means a
        # resumed run replays the same task sequence from the start, which in the
        # first 20-iteration run made iterations 10-19 draw exactly the tasks
        # 0-9 had drawn.
        picked = random.Random(args.seed * 10_000 + it).sample(
            frontier, min(args.prompts_per_iter, len(frontier)))
        keys = [(t["level"], t["problem_id"]) for t in picked]

        # --- rollout ------------------------------------------------------
        messages, owner = [], []
        for key in keys:
            for _ in range(args.group_size):
                messages.append([{"role": "user", "content": prompts[key]}])
                owner.append(key)
        t0 = time.time()
        texts = sample_batch(chat, messages, args.concurrency)
        t_roll = time.time() - t0

        # --- reward -------------------------------------------------------
        items = []
        for j, (key, text) in enumerate(zip(owner, texts)):
            code = extract_code(text) if not text.startswith("__ERROR__") else None
            items.append(("%d" % j, code, refs[key]))
        t0 = time.time()
        scored = score_rollouts(items, gpus=args.gpus, workers=workers,
                                measure_speed=not args.no_speed, pool=pool)
        t_reward = time.time() - t0
        stats = summarise(scored)
        n_inconclusive = sum(1 for r, rec in scored.values()
                             if r is None or rec.get("stage") in
                             ("oom", "cuda_poison", "worker_crash"))
        if scored and n_inconclusive == len(scored):
            raise SystemExit(
                "iter %d: all %d rollouts were inconclusive (verifier CUDA "
                "init failed). Not writing skip records that would look like "
                "finished iterations." % (it, len(scored)))

        # --- advantages ---------------------------------------------------
        by_group = {}
        for j, key in enumerate(owner):
            by_group.setdefault(key, []).append(j)
        adv = [0.0] * len(owner)
        n_live_groups = 0
        for key, idxs in by_group.items():
            rewards = [scored["%d" % j][0] for j in idxs]
            a = group_advantages(rewards)
            if any(abs(x) > 1e-9 for x in a):
                n_live_groups += 1
            for j, x in zip(idxs, a):
                adv[j] = x

        # Only sequences with a nonzero advantage and extractable code can teach.
        train_idx = [j for j in range(len(owner))
                     if abs(adv[j]) > 1e-9 and items[j][1]]

        # Degeneration to empty output shows up here first, and the previous
        # version returned before writing any history -- so the five iterations
        # over which a collapse actually happened left no trace at all, and the
        # log jumped straight from healthy to dead. Record the skip, then decide.
        if stats["no_code_rate"] > args.max_no_code or not train_idx:
            rec = {"iteration": it, "prompts": len(keys),
                   "rollouts": len(owner), "live_groups": n_live_groups,
                   "trained_on": 0, "skipped": True,
                   "mean_reward": round(stats["mean_reward"], 4),
                   "pass_rate": round(stats["pass_rate"], 4),
                   "purity_rate": round(stats["purity_rate"], 4),
                   "fast_rate": round(stats["fast_rate"], 4),
                   "no_code_rate": round(stats["no_code_rate"], 4)}
            history.append(rec)
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if stats["no_code_rate"] > args.max_no_code:
                raise SystemExit(
                    "iter %d: %.1f%% of rollouts produced no code (limit %.0f%%). "
                    "The policy is degenerating; stopping so the snapshots stay "
                    "usable. Lower --lr or raise --kl-coef."
                    % (it, stats["no_code_rate"] * 100, args.max_no_code * 100))
            print("iter %d: no group had any spread; skipping the update" % it)
            continue

        # --- policy gradient ----------------------------------------------
        t0 = time.time()
        seqs = []
        n_overflow = 0
        for j in train_idx:
            # Score the sampled text, not extract_code(text). The kernel fence
            # is what the reward saw; the tokens the policy emitted include the
            # reasoning channel.
            sampled = texts[j]
            if sampled.startswith("__ERROR__"):
                continue
            ids, mask = build_sequence(
                tok, prompts[owner[j]], sampled, args.max_len,
                chat_kwargs=chat_kwargs, turn_end=turn_end)
            if ids is None or sum(mask) < 2:
                n_overflow += 1
                continue
            seqs.append((j, ids, mask))

        if not seqs:
            rec = {"iteration": it, "prompts": len(keys),
                   "rollouts": len(owner), "live_groups": n_live_groups,
                   "trained_on": 0, "skipped": True, "overflow": n_overflow,
                   "mean_reward": round(stats["mean_reward"], 4),
                   "pass_rate": round(stats["pass_rate"], 4),
                   "purity_rate": round(stats["purity_rate"], 4),
                   "fast_rate": round(stats["fast_rate"], 4),
                   "no_code_rate": round(stats["no_code_rate"], 4)}
            history.append(rec)
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("iter %d: %d live rollouts overflowed max_len=%d; skipping"
                  % (it, n_overflow, args.max_len))
            continue

        n_clipped = n_tok = 0
        losses, kls = [], []
        opt.zero_grad(set_to_none=True)
        done_micro = 0
        for i, (j, ids, mask) in enumerate(seqs):
            t_ids = torch.tensor([ids], device=dev)
            t_mask = torch.tensor([mask], device=dev)
            # At one inner epoch the ratio is exactly 1 on the only pass that
            # happens, so the reference forward produces a constant and the
            # objective reduces to REINFORCE with a group baseline. Computing it
            # anyway doubled the cost of every step in the first run.
            old = None
            if args.inner_epochs > 1:
                old = completion_logprobs(model, t_ids, t_mask, grad=False)
                if old is None:
                    continue
                old = old.detach()

            # The starting policy, obtained by holding LoRA scaling at 0 so the
            # compute path matches the trainable adapter. Costs one forward
            # rather than a second copy of the weights.
            ref = None
            if args.kl_coef > 0:
                with adapter_kept_at_base(model):
                    ref = completion_logprobs(model, t_ids, t_mask, grad=False)
                if ref is None:
                    continue
                ref = ref.detach()
                if args.fresh_lora and it == start_iter and i == 0:
                    with torch.no_grad():
                        on = completion_logprobs(model, t_ids, t_mask, grad=False)
                    if on is not None:
                        seq_l1 = float((on - ref).abs().mean())
                        on2 = completion_logprobs(model, t_ids, t_mask,
                                                  grad=False)
                        same_l1 = (float((on - on2).abs().mean())
                                   if on2 is not None else float("nan"))
                        print("iter %d seq0 scaling-0 L1=%.6f two-forward L1=%.6f "
                              "(tokens %d)"
                              % (it, seq_l1, same_l1, t_ids.shape[1]))
                        # Two-forward noise is not a broken adapter; only refuse
                        # when the reference path disagrees beyond that floor.
                        floor = (0.05 if math.isnan(same_l1)
                                 else max(0.05, 10.0 * same_l1 + 0.02))
                        if seq_l1 > floor:
                            raise SystemExit(
                                "scaling-0 L1 %.4f on a real sequence "
                                "(fresh LoRA must be identity; two-forward "
                                "L1 %.4f); refusing."
                                % (seq_l1, same_l1))

            for _ in range(args.inner_epochs):
                new = completion_logprobs(model, t_ids, t_mask, grad=True)
                if new is None:
                    break
                a = adv[j]
                if old is None:
                    loss = -(a * new).mean()
                    n_tok += new.numel()
                else:
                    ratio = torch.exp(new - old)
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - args.clip_eps,
                                          1 + args.clip_eps) * a
                    loss = -torch.min(unclipped, clipped).mean()
                    n_clipped += int((unclipped > clipped).sum())
                    n_tok += ratio.numel()
                if ref is not None:
                    # k3 estimator: non-negative and lower variance than the
                    # plain log-ratio difference.
                    d = ref - new
                    kl = (torch.exp(d) - d - 1).mean()
                    loss = loss + args.kl_coef * kl
                    kls.append(kl.item())
                    if args.fresh_lora and it == start_iter and kl.item() > 1.0:
                        raise SystemExit(
                            "iter %d KL %.1f on a fresh LoRA (expected ~0). "
                            "Refusing to write this adapter." % (it, kl.item()))
                if not torch.isfinite(loss):
                    break
                (loss / args.grad_accum).backward()
                losses.append(loss.item())
                done_micro += 1
                if done_micro % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        if done_micro % args.grad_accum:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        t_train = time.time() - t0

        rec = {
            "iteration": it,
            "prompts": len(keys), "rollouts": len(owner),
            "live_groups": n_live_groups, "trained_on": len(seqs),
            "overflow": n_overflow,
            "mean_reward": round(stats["mean_reward"], 4),
            "pass_rate": round(stats["pass_rate"], 4),
            "purity_rate": round(stats["purity_rate"], 4),
            "fast_rate": round(stats["fast_rate"], 4),
            "no_code_rate": round(stats["no_code_rate"], 4),
            "clip_frac": round(n_clipped / max(n_tok, 1), 4),
            "kl": round(sum(kls) / max(len(kls), 1), 5),
            "loss": round(sum(losses) / max(len(losses), 1), 5),
            "seconds": {"rollout": round(t_roll), "reward": round(t_reward),
                        "train": round(t_train), "total": round(time.time() - t_iter)},
        }
        history.append(rec)
        # Each iteration draws different tasks, so a single iteration's reward
        # says more about which tasks it drew than about the policy. The window
        # is what to read for a trend; the honest verdict is the held-out
        # evaluation at the end.
        window = [h["mean_reward"] for h in history[-5:]]
        rec["reward_w5"] = round(sum(window) / len(window), 4)
        print("iter %2d  reward %.3f (w5 %.3f)  pass %.3f  pure %.3f  fast %.3f  "
              "nocode %.3f  groups %d/%d  kl %.4f  (%.0fs roll, %.0fs reward, "
              "%.0fs train)"
              % (it, rec["mean_reward"], rec["reward_w5"], rec["pass_rate"],
                 rec["purity_rate"], rec["fast_rate"], rec["no_code_rate"],
                 n_live_groups, len(keys), rec["kl"], t_roll, t_reward, t_train))

        # --- checkpoint -----------------------------------------------------
        # Only the trainable tensors: everything else is unchanged from the
        # starting adapter, and writing the frozen experts too would put a
        # 27.5 GB write in the inner loop of a run on a machine that expires.
        ck = os.path.join(args.out, "ck")
        os.makedirs(ck, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                    if k in {n for n, p in model.named_parameters()
                             if p.requires_grad}},
                   os.path.join(ck, "trainable.pt"))
        torch.save({"optimizer": opt.state_dict(), "iteration": it,
                    "args": vars(args)}, os.path.join(ck, "opt.pt"))
        # Also write a loadable adapter. Without it the deltas have to be
        # replayed onto the starting adapter by hand before anything can serve
        # or evaluate the policy, and the refresh loop needs to do that every
        # few iterations.
        model.save_pretrained(os.path.join(ck, "adapter"))
        # A rolling checkpoint is not a safety net: the collapse in the first run
        # overwrote the healthy weights with degenerate ones on every iteration,
        # so there was nothing to fall back to. Numbered snapshots are cheap at
        # 526 MB.
        if args.snapshot_every and (it + 1) % args.snapshot_every == 0:
            model.save_pretrained(os.path.join(args.out, "snap-%03d" % it))
        with open(os.path.join(args.out, "history.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    print("\ndone. checkpoint in %s/ck; merge with train/merge_lora.py after "
          "applying the deltas to the starting adapter" % args.out)
    pool.close()


if __name__ == "__main__":
    main()
