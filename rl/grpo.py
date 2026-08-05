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

No explicit KL penalty. The only free reference policy is the base model with the
adapter switched off, and pulling toward that would undo the supervised rounds
this run starts from; holding a frozen copy of the current adapter instead would
cost another 27.5 GB. Clipping plus a small number of inner epochs keeps the
policy close enough, and purity rate and output length are logged as the
degeneration alarms.

Only the attention and DeltaNet adapters are trained. The expert parameters are
frozen, which drops the trainable count from 6.88B to about 34M and the
checkpoint from 27.5 GB to under 100 MB -- the difference between checkpointing
every iteration and not being able to afford it. See train/lora_config.py.

Usage:
    python3 rl/grpo.py --model /raid/... --adapter models/lora-C-speed \\
        --frontier runs/rl_frontier.json --out runs/grpo --iterations 4
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "repair"))
sys.path.insert(0, HERE)

from lora_config import freeze_experts  # noqa: E402
from repair_loop import Chat, extract_code, sample_batch  # noqa: E402
from reward import score_rollouts, summarise  # noqa: E402
from train_lora import unwrap  # noqa: E402


def build_sequence(tok, prompt: str, completion: str, max_len: int):
    """Tokenise a rollout the way training saw it, and mark the completion.

    Must match CompletionOnlyDataset: the same chat template, the same eos, and
    truncation from the front of the prompt. A mismatch here would have the model
    scoring token positions it never actually generated.
    """
    from train_lora import CompletionOnlyDataset

    prompt_ids = CompletionOnlyDataset._as_ids(tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True))
    completion_ids = tok(completion + tok.eos_token,
                         add_special_tokens=False)["input_ids"]

    ids = prompt_ids + completion_ids
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    if len(ids) > max_len:
        overflow = len(ids) - max_len
        ids, mask = ids[overflow:], mask[overflow:]
    return ids, mask


def completion_logprobs(model, ids, comp_mask, grad: bool):
    """Per-token log probability of the completion tokens.

    Applies the LM head only at the positions being scored. The prompt is ~14k
    tokens of documentation and the completion is a fraction of that, so
    projecting everything through a 152k vocabulary would cost gigabytes to
    produce numbers that are immediately discarded.
    """
    body, lm_head = unwrap(model)
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
        logits = lm_head(h[keep]).float()
        return torch.log_softmax(logits, dim=-1).gather(
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base weights.")
    ap.add_argument("--adapter", required=True, help="Starting policy adapter.")
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
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=17408)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--served-model", default="Qwen3-Coder-Next")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--verify-workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--no-speed", action="store_true",
                    help="Skip timing; correctness-only reward.")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint directory from an earlier invocation.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    frontier = json.load(open(args.frontier))
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
            "cutile_docs", ref_arch_src=problem.code, backend="cutile",
            option="one_shot", precision="fp32")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("loading policy ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    print("loaded in %.0fs" % (time.time() - t0))

    trainable, frozen = freeze_experts(model)
    print("training %s of %s parameters (%.3f%%); experts frozen"
          % ("{:,}".format(trainable), "{:,}".format(trainable + frozen),
             trainable / (trainable + frozen) * 100))
    if trainable == 0:
        raise SystemExit("nothing left trainable after freezing experts")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    model.config.use_cache = False

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), foreach=False)

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
    dev = next(model.parameters()).device

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
        scored = score_rollouts(items, gpus=args.gpus, workers=args.verify_workers,
                                measure_speed=not args.no_speed)
        t_reward = time.time() - t0
        stats = summarise(scored)

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
        if not train_idx:
            print("iter %d: no group had any spread; skipping the update" % it)
            continue

        # --- policy gradient ----------------------------------------------
        t0 = time.time()
        seqs = []
        for j in train_idx:
            ids, mask = build_sequence(tok, prompts[owner[j]], items[j][1],
                                       args.max_len)
            if sum(mask) < 2:
                continue
            seqs.append((j, ids, mask))

        n_clipped = n_tok = 0
        losses = []
        opt.zero_grad(set_to_none=True)
        done_micro = 0
        for i, (j, ids, mask) in enumerate(seqs):
            t_ids = torch.tensor([ids], device=dev)
            t_mask = torch.tensor([mask], device=dev)
            old = completion_logprobs(model, t_ids, t_mask, grad=False)
            if old is None:
                continue
            old = old.detach()

            for _ in range(args.inner_epochs):
                new = completion_logprobs(model, t_ids, t_mask, grad=True)
                if new is None:
                    break
                ratio = torch.exp(new - old)
                a = adv[j]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - args.clip_eps,
                                      1 + args.clip_eps) * a
                loss = -torch.min(unclipped, clipped).mean()
                if not torch.isfinite(loss):
                    break
                (loss / args.grad_accum).backward()
                losses.append(loss.item())
                n_clipped += int((unclipped > clipped).sum())
                n_tok += ratio.numel()
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
            "mean_reward": round(stats["mean_reward"], 4),
            "pass_rate": round(stats["pass_rate"], 4),
            "purity_rate": round(stats["purity_rate"], 4),
            "fast_rate": round(stats["fast_rate"], 4),
            "no_code_rate": round(stats["no_code_rate"], 4),
            "clip_frac": round(n_clipped / max(n_tok, 1), 4),
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
              "groups %d/%d  clip %.3f  (%.0fs roll, %.0fs reward, %.0fs train)"
              % (it, rec["mean_reward"], rec["reward_w5"], rec["pass_rate"],
                 rec["purity_rate"], rec["fast_rate"], n_live_groups, len(keys),
                 rec["clip_frac"], t_roll, t_reward, t_train))

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
        with open(os.path.join(args.out, "history.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    print("\ndone. checkpoint in %s/ck; merge with train/merge_lora.py after "
          "applying the deltas to the starting adapter" % args.out)


if __name__ == "__main__":
    main()
