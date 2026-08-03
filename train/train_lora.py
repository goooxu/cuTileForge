"""LoRA SFT on verifier-accepted cuTile kernels.

Loss is computed on the completion only. The prompt carries ~14k tokens of
cuTile documentation that is identical across examples, and training the model
to reproduce it would waste the entire gradient budget on a fixed prefix.

Target modules come from lora_config, which resolves the Gated DeltaNet
projection names against the actual model; see the note there about why the
usual attention-only names are not enough for this architecture.

Usage inside the training image:
    python3 train/train_lora.py --model /ws/models/Qwen3-Coder-Next \\
        --data /ws/runs/sft_l92.jsonl --out /ws/models/lora-cutile
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_config import DEFAULT_TARGETS, DELTANET_TARGETS, resolve_targets  # noqa: E402


class CompletionOnlyDataset(Dataset):
    """Tokenised prompt+completion with the prompt masked out of the loss."""

    def __init__(self, path, tokenizer, max_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.max_len = max_len
        self.n_truncated = 0

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _as_ids(x):
        """Normalise apply_chat_template output to a flat list of token ids.

        transformers 5.x returns a BatchEncoding here rather than a list, and
        may nest the ids one level deep depending on the call.
        """
        if hasattr(x, "input_ids"):
            x = x.input_ids
        elif isinstance(x, dict):
            x = x["input_ids"]
        if x and isinstance(x[0], (list, tuple)):
            x = x[0]
        return list(x)

    def __getitem__(self, i):
        row = self.rows[i]
        # Match how the model is prompted at inference: a user turn through the
        # chat template, with the assistant turn as the target.
        prompt_ids = self._as_ids(self.tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
            tokenize=True,
        ))
        completion_ids = self.tok(row["completion"] + self.tok.eos_token,
                                  add_special_tokens=False)["input_ids"]

        ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)

        if len(ids) > self.max_len:
            # Trim from the front of the prompt: the completion and the task
            # description at the prompt's end are what matter.
            overflow = len(ids) - self.max_len
            ids = ids[overflow:]
            labels = labels[overflow:]
            self.n_truncated += 1

        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=20480)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    ds = CompletionOnlyDataset(args.data, tok, args.max_len)
    print("dataset: %d examples" % len(ds))

    print("loading model ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    print("loaded in %.1fs" % (time.time() - t0))

    counts, _, _ = resolve_targets(model, DEFAULT_TARGETS)
    missing = [t for t, c in counts.items() if c == 0]
    if missing:
        raise SystemExit("target modules match nothing: %s" % ", ".join(missing))
    if not any(counts[t] for t in DELTANET_TARGETS):
        raise SystemExit("no DeltaNet projections matched")

    # lora_dropout must stay 0: some of these targets are fused MoE parameters
    # rather than nn.Linear, and peft wraps those with ParamWrapper, which
    # rejects any nonzero dropout.
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=DEFAULT_TARGETS))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable %s (%.3f%%)"
          % (f"{trainable:,}",
             trainable / sum(p.numel() for p in model.parameters()) * 100))

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    model.config.use_cache = False

    # Batch size 1 with accumulation: sequences are ~15-20k tokens, so a single
    # example already fills the activation budget.
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_frac, anneal_strategy="cos")

    print("optimiser steps: %d (%d micro-batches/epoch, accum %d)"
          % (total_steps, len(loader), args.grad_accum))

    dev = next(model.parameters()).device
    step = 0
    micro = 0
    n_skipped = 0
    running = []
    history = []
    t_start = time.time()
    done = False

    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for input_ids, labels, attn in loader:
            n_labels = int((labels != -100).sum())
            out = model(input_ids=input_ids.to(dev),
                        attention_mask=attn.to(dev),
                        labels=labels.to(dev))
            loss = out.loss
            if not torch.isfinite(loss):
                # A single bad example should not end the run. Drop its
                # contribution and keep going, but refuse to train on garbage if
                # it turns out to be widespread.
                n_skipped += 1
                print("  skipped micro-batch %d: non-finite loss "
                      "(seq %d tokens, %d labels)"
                      % (micro, input_ids.shape[1], n_labels))
                opt.zero_grad(set_to_none=True)
                micro += 1
                if n_skipped > max(10, 0.05 * len(loader)):
                    raise SystemExit("too many non-finite losses (%d); aborting"
                                     % n_skipped)
                continue
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            micro += 1
            if micro <= 4:
                print("  micro %d: loss %.4f (seq %d, labels %d)"
                      % (micro - 1, loss.item(), input_ids.shape[1], n_labels))

            if micro % args.grad_accum == 0:
                if not running:
                    # Every micro-batch in this window was skipped.
                    opt.zero_grad(set_to_none=True)
                    continue
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0 or step == 1:
                    mean = sum(running) / len(running)
                    history.append({"step": step, "loss": mean})
                    print("step %4d/%d  loss %.4f  gnorm %.3f  lr %.2e  %.1f min"
                          % (step, total_steps, mean, gnorm,
                             sched.get_last_lr()[0], (time.time() - t_start) / 60))
                    running = []

                if step >= total_steps:
                    done = True
                    break

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"history": history, "args": vars(args),
                   "truncated_examples": ds.n_truncated}, f, indent=2)

    print("\ntruncated %d/%d examples at max_len=%d"
          % (ds.n_truncated, len(ds), args.max_len))
    print("skipped %d micro-batches for non-finite loss" % n_skipped)
    print("peak GPU memory: %.1f GB"
          % (max(torch.cuda.max_memory_allocated(i)
                 for i in range(torch.cuda.device_count())) / 1e9))
    print("saved adapter to", args.out)


if __name__ == "__main__":
    main()
