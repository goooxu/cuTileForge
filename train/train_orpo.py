"""LoRA ORPO on within-problem kernel_ms pairs.

Chosen is the lowest-latency correct harvest trace, rejected the highest.
The SFT term is only on chosen (same completion-only loss as train_lora.py).
The odds-ratio term is applied in a second backward so peak memory stays
one sequence, not a 2x graph. Optional --retain adds a third backward of
completion-only SFT (conv without a GEMM tile) so matmul tiles do not leak.

  python3 train/train_orpo.py --model /raid/tmp/gemsg-cutile/model-GLE \
      --data /ws/runs/orpo_glh.jsonl --retain /ws/runs/sft_gli_retain.jsonl \
      --out /ws/models/lora-GLI
"""
from __future__ import print_function

import argparse
import collections
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_config import (family_of, load_base_model,  # noqa: E402
                         targets_for, validate_targets)
from train_lora import (  # noqa: E402
    TURN_END, CompletionOnlyDataset, collate as collate_sft, sparse_lm_loss,
)


class PairDataset(Dataset):
    """One prompt, two completions. Drop the pair if either side overflows."""

    def __init__(self, path, tokenizer, max_len, chat_kwargs=None,
                 turn_end=None):
        self.max_len = max_len
        self.helper = CompletionOnlyDataset.__new__(CompletionOnlyDataset)
        self.helper.tok = tokenizer
        self.helper.max_len = max_len
        self.helper.chat_kwargs = dict(chat_kwargs or {})
        self.helper.turn_end = (
            turn_end if turn_end is not None else tokenizer.eos_token)
        self.helper.on_overflow = "drop"
        self.turn_end = self.helper.turn_end
        self.n_dropped = 0
        raw = [json.loads(l) for l in open(path) if l.strip()]
        self.rows = []
        for row in raw:
            if self._fits(row):
                self.rows.append(row)
            else:
                self.n_dropped += 1

    def _fits(self, row):
        for key in ("chosen", "rejected"):
            ids, _, _ = self.helper._encode({
                "prompt": row["prompt"], "completion": row[key],
            })
            if len(ids) > self.max_len:
                return False
        return True

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        out = {"category": row.get("category", "?")}
        for key in ("chosen", "rejected"):
            ids, n_prompt, cids = self.helper._encode({
                "prompt": row["prompt"], "completion": row[key],
            })
            out[key] = {
                "input_ids": ids,
                "labels": [-100] * n_prompt + list(cids),
            }
        return out


def collate_pair(batch, pad_id):
    def pack(side):
        width = max(len(b[side]["input_ids"]) for b in batch)
        ids, labels, attn = [], [], []
        for b in batch:
            pad = width - len(b[side]["input_ids"])
            ids.append(b[side]["input_ids"] + [pad_id] * pad)
            labels.append(b[side]["labels"] + [-100] * pad)
            attn.append([1] * len(b[side]["input_ids"]) + [0] * pad)
        return torch.tensor(ids), torch.tensor(labels), torch.tensor(attn)

    return pack("chosen"), pack("rejected"), [b["category"] for b in batch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orpo-lambda", type=float, default=0.5)
    ap.add_argument("--retain", default=None,
                    help="Optional SFT jsonl mixed 1:1 with each ORPO pair. "
                         "Use a no-MMA conv retain set so matmul tiles do not "
                         "leak into conv.")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--max-skip-frac", type=float, default=0.05)
    ap.add_argument("--targets", default="attention_only",
                    choices=["default", "attention_only"])
    ap.add_argument("--reasoning-strength", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoConfig, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    family = family_of(AutoConfig.from_pretrained(args.model,
                                                  trust_remote_code=True))
    chat_kwargs = ({"reasoning_strength": args.reasoning_strength}
                   if args.reasoning_strength else {})
    ds = PairDataset(args.data, tok, args.max_len,
                     chat_kwargs=chat_kwargs,
                     turn_end=TURN_END.get(family))
    print("dataset: %d pairs (dropped %d over max_len=%d, turn end %r)"
          % (len(ds), ds.n_dropped, args.max_len, ds.turn_end))
    if not len(ds):
        raise SystemExit("no pairs fit in max_len=%d" % args.max_len)

    retain_iter = None
    n_retain = 0
    if args.retain:
        retain_ds = CompletionOnlyDataset(
            args.retain, tok, args.max_len,
            chat_kwargs=chat_kwargs, turn_end=ds.turn_end)
        n_retain = len(retain_ds)
        print("retain: %d rows (dropped %d over max_len=%d) from %s"
              % (n_retain, retain_ds.n_dropped, args.max_len, args.retain))
        if not n_retain:
            raise SystemExit("retain set empty after max_len filter")
        retain_loader = DataLoader(
            retain_ds, batch_size=1, shuffle=True,
            collate_fn=lambda b: collate_sft(b, pad_id))

        def _cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        retain_iter = _cycle(retain_loader)

    print("loading model ...")
    t0 = time.time()
    model = load_base_model(args.model, device_map="auto",
                            dtype=torch.bfloat16)
    print("loaded in %.1fs" % (time.time() - t0))

    targets = targets_for(model, args.targets)
    validate_targets(model, targets)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable %s (%.3f%%)"
          % (f"{trainable:,}",
             trainable / sum(p.numel() for p in model.parameters()) * 100))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()
    model.config.use_cache = False

    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=lambda b: collate_pair(b, pad_id))
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), foreach=False)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_frac, anneal_strategy="cos")
    print("optimiser steps: %d (%d pairs/epoch, accum %d, lambda %.2f%s)"
          % (total_steps, len(loader), args.grad_accum, args.orpo_lambda,
             (", retain %d 1:1" % n_retain) if n_retain else ""))

    dev = next(model.parameters()).device
    step = micro = n_skipped = 0
    seen_cats = collections.Counter()
    skipped_cats = collections.Counter()
    running = []
    history = []
    t_start = time.time()
    done = False

    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for chosen, rejected, cats in loader:
            seen_cats[cats[0]] += 1
            c_ids, c_lab, c_attn = [x.to(dev) for x in chosen]
            r_ids, r_lab, r_attn = [x.to(dev) for x in rejected]
            nll_c = sparse_lm_loss(model, c_ids, c_attn, c_lab)
            if not torch.isfinite(nll_c):
                n_skipped += 1
                skipped_cats[cats[0]] += 1
                if n_skipped <= 10 or n_skipped % 25 == 0:
                    print("  skipped pair %d: non-finite chosen nll [%d skipped]"
                          % (micro, n_skipped))
                opt.zero_grad(set_to_none=True)
                micro += 1
                if n_skipped > max(10, args.max_skip_frac * len(loader)):
                    raise SystemExit("too many non-finite losses (%d); aborting"
                                     % n_skipped)
                continue
            nll_c_val = nll_c.detach()
            (nll_c / args.grad_accum).backward()

            nll_r = sparse_lm_loss(model, r_ids, r_attn, r_lab)
            if not torch.isfinite(nll_r):
                n_skipped += 1
                skipped_cats[cats[0]] += 1
                if n_skipped <= 10 or n_skipped % 25 == 0:
                    print("  skipped pair %d: non-finite rejected nll [%d skipped]"
                          % (micro, n_skipped))
                opt.zero_grad(set_to_none=True)
                micro += 1
                if n_skipped > max(10, args.max_skip_frac * len(loader)):
                    raise SystemExit("too many non-finite losses (%d); aborting"
                                     % n_skipped)
                continue
            # Detached chosen NLL: one sequence in the graph at a time.
            # Chosen still received the SFT gradient above.
            or_loss = -F.logsigmoid(nll_r - nll_c_val)
            ((args.orpo_lambda * or_loss) / args.grad_accum).backward()
            nll_retain = None
            if retain_iter is not None:
                s_ids, s_lab, s_attn, _scats = next(retain_iter)
                s_ids, s_lab, s_attn = s_ids.to(dev), s_lab.to(dev), s_attn.to(dev)
                nll_s = sparse_lm_loss(model, s_ids, s_attn, s_lab)
                if torch.isfinite(nll_s):
                    (nll_s / args.grad_accum).backward()
                    nll_retain = float(nll_s.detach())
            loss = nll_c_val + args.orpo_lambda * or_loss.detach()
            rec = {
                "loss": float(loss.detach()), "nll_c": float(nll_c_val),
                "nll_r": float(nll_r.detach()), "or": float(or_loss.detach()),
            }
            if nll_retain is not None:
                rec["nll_retain"] = nll_retain
            running.append(rec)
            micro += 1
            if micro <= 4:
                print("  micro %d: loss %.4f  nll_c %.4f  nll_r %.4f  or %.4f"
                      % (micro - 1, loss.item(), nll_c.item(),
                         nll_r.item(), or_loss.item()))

            if micro % args.grad_accum == 0:
                if not running:
                    opt.zero_grad(set_to_none=True)
                    continue
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == 1:
                    mean = sum(x["loss"] for x in running) / len(running)
                    mean_c = sum(x["nll_c"] for x in running) / len(running)
                    mean_r = sum(x["nll_r"] for x in running) / len(running)
                    extra = ""
                    retain_vals = [x["nll_retain"] for x in running
                                   if "nll_retain" in x]
                    if retain_vals:
                        extra = "  nll_ret %.4f" % (
                            sum(retain_vals) / len(retain_vals))
                    history.append({"step": step, "loss": mean,
                                    "nll_chosen": mean_c, "nll_rejected": mean_r})
                    print("step %4d/%d  loss %.4f  nll_c %.4f  nll_r %.4f%s  "
                          "gnorm %.3f  lr %.2e  %.1f min"
                          % (step, total_steps, mean, mean_c, mean_r, extra,
                             gnorm, sched.get_last_lr()[0],
                             (time.time() - t_start) / 60))
                    running = []
                if step >= total_steps:
                    done = True
                    break

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"history": history, "args": vars(args),
                   "dropped_pairs": ds.n_dropped}, f, indent=2)
    print("skipped %d pairs for non-finite loss" % n_skipped)
    print("peak GPU memory: %.1f GB"
          % (max(torch.cuda.max_memory_allocated(i)
                 for i in range(torch.cuda.device_count())) / 1e9))
    if n_skipped:
        print("dropped %d/%d pairs (%.1f%%) to non-finite loss:"
              % (n_skipped, micro, n_skipped / max(micro, 1) * 100))
        for cat in sorted(seen_cats, key=lambda c: -seen_cats[c]):
            n_seen, n_drop = seen_cats[cat], skipped_cats.get(cat, 0)
            print("    %-12s %4d seen, %4d dropped (%4.1f%%)"
                  % (cat, n_seen, n_drop, n_drop / max(n_seen, 1) * 100))
    print("saved adapter to", args.out)


if __name__ == "__main__":
    main()
