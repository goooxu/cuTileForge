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
import collections
import json
import math
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_config import (ATTENTION_ONLY_TARGETS, DEFAULT_TARGETS,  # noqa: E402
                         DELTANET_TARGETS, resolve_targets)


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

        # Category rides along so the trainer can report whether dropped
        # micro-batches are concentrated in one operator family.
        return {"input_ids": ids, "labels": labels,
                "category": row.get("category", "?")}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn),
            [b.get("category", "?") for b in batch])


def unwrap(model):
    """Reach the transformer body and the LM head through PEFT's wrappers.

    PeftModel forwards unknown attributes to the model it wraps, so probing with
    hasattr finds lm_head on the wrapper itself and stops one level too high --
    .model there is still the causal LM, not the body. Descend explicitly
    instead. LoRA layers are injected into the wrapped modules, so going through
    the base model still applies the adapter.
    """
    m = model
    while hasattr(m, "get_base_model"):
        m = m.get_base_model()
    if not (hasattr(m, "lm_head") and hasattr(m, "model")):
        raise SystemExit("expected a causal LM with .model and .lm_head, got %s"
                         % type(m).__name__)
    return m.model, m.lm_head


def sparse_lm_loss(model, input_ids, attn, labels):
    """Cross-entropy over only the positions that carry a label.

    Passing labels= to the model projects every position through the LM head,
    which at a 152k vocabulary and 16k tokens is a 5 GB logits tensor before the
    fp32 upcast that cross-entropy does. Prompts here are ~14k tokens of fixed
    documentation and only the completion is supervised, so fewer than one
    position in ten carries a label. Selecting first makes that tensor two orders
    of magnitude smaller, which is most of the difference between a run that
    fits in 87 GB and one that does not fit at all. The value is identical: the
    mean is over the same tokens, and test_sparse_loss.py checks that against
    the model's own labels= path.
    """
    import torch.nn.functional as F

    body, lm_head = unwrap(model)
    hidden = body(input_ids=input_ids, attention_mask=attn).last_hidden_state

    # Standard causal shift: position t predicts token t+1.
    hidden = hidden[:, :-1]
    # The model is sharded, so the last layer's output need not be on the device
    # the inputs were placed on.
    target = labels[:, 1:].to(hidden.device)
    keep = target != -100
    if not keep.any():
        return torch.tensor(float("nan"), device=hidden.device)
    return F.cross_entropy(lm_head(hidden[keep]).float(), target[keep])


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
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Trade compute for activation memory. Required to fit, but "
                         "makes some losses NaN; see the comment at the call site.")
    ap.add_argument("--max-skip-frac", type=float, default=0.05,
                    help="Abort if more than this fraction of micro-batches have a "
                         "non-finite loss.")
    ap.add_argument("--targets", default="default",
                    choices=["default", "attention_only"],
                    help="default reaches the routed experts through peft's "
                         "ParamWrapper, which trains 6.88B including a full "
                         "fine-tune of their base weights. attention_only keeps "
                         "to the attention and DeltaNet projections, where a much "
                         "higher rank is affordable. See train/lora_config.py.")
    ap.add_argument("--resume-adapter", default=None,
                    help="Continue training an existing adapter instead of a fresh "
                         "one. Note this stacks rounds: the result inherits whatever "
                         "the earlier round's data distribution taught it.")
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

    targets = (ATTENTION_ONLY_TARGETS if args.targets == "attention_only"
               else DEFAULT_TARGETS)
    counts, _, _ = resolve_targets(model, targets)
    missing = [t for t, c in counts.items() if c == 0]
    if missing:
        raise SystemExit("target modules match nothing: %s" % ", ".join(missing))
    if not any(counts[t] for t in DELTANET_TARGETS):
        raise SystemExit("no DeltaNet projections matched")

    if args.resume_adapter:
        from peft import PeftModel
        print("resuming from adapter", args.resume_adapter)
        # is_trainable is required: without it peft loads the adapter frozen for
        # inference and the run would silently optimise nothing.
        model = PeftModel.from_pretrained(model, args.resume_adapter,
                                          is_trainable=True)
    else:
        # lora_dropout must stay 0: some of these targets are fused MoE parameters
        # rather than nn.Linear, and peft wraps those with ParamWrapper, which
        # rejects any nonzero dropout.
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable %s (%.3f%%)"
          % (f"{trainable:,}",
             trainable / sum(p.numel() for p in model.parameters()) * 100))

    # A genuine trade, not a free win. Checkpointing is what makes losses go
    # non-finite here: with it off, 1-2% of micro-batches are dropped, with it on,
    # 20-30%, on the same data. But with it off activations exceed 180 GB on the
    # first GPU and the run dies partway through an epoch, and that survived every
    # attempt to economise -- selecting labelled positions, foreach=False,
    # expandable_segments, a lower max_len -- each of which only moved the OOM
    # later. So: keep checkpointing, drop the bad micro-batches. That is only
    # acceptable while the drops stay spread across operator families, which the
    # summary at the end of the run reports rather than assumes.
    if args.gradient_checkpointing:
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
    # foreach=False costs some step throughput but avoids the multi-tensor
    # temporaries. Kept from the attempt to train without checkpointing, where
    # the optimiser step was where memory tipped over; harmless now that
    # checkpointing is back on, and useful again if it ever has to come off.
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), foreach=False)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_frac, anneal_strategy="cos")

    print("optimiser steps: %d (%d micro-batches/epoch, accum %d)"
          % (total_steps, len(loader), args.grad_accum))

    dev = next(model.parameters()).device
    step = 0
    micro = 0
    n_skipped = 0
    seen_cats = collections.Counter()
    skipped_cats = collections.Counter()
    running = []
    history = []
    t_start = time.time()
    done = False

    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for input_ids, labels, attn, cats in loader:
            n_labels = int((labels != -100).sum())
            seen_cats[cats[0]] += 1
            loss = sparse_lm_loss(model, input_ids.to(dev), attn.to(dev),
                                  labels.to(dev))
            if not torch.isfinite(loss):
                # A single bad example should not end the run. Drop its
                # contribution and keep going, but refuse to train on garbage if
                # it turns out to be widespread.
                n_skipped += 1
                skipped_cats[cats[0]] += 1
                if n_skipped <= 10 or n_skipped % 25 == 0:
                    print("  skipped micro-batch %d: non-finite loss "
                          "(seq %d tokens, %d labels) [%d skipped so far]"
                          % (micro, input_ids.shape[1], n_labels, n_skipped))
                opt.zero_grad(set_to_none=True)
                micro += 1
                if n_skipped > max(10, args.max_skip_frac * len(loader)):
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

    if n_skipped:
        # Dropping micro-batches is only acceptable if it is spread across
        # operator families. A category-specific drop rate would quietly recreate
        # the coverage gap this dataset was rebuilt to close.
        print("dropped %d/%d micro-batches (%.1f%%) to non-finite loss:"
              % (n_skipped, micro, n_skipped / max(micro, 1) * 100))
        for cat in sorted(seen_cats, key=lambda c: -seen_cats[c]):
            n_seen, n_drop = seen_cats[cat], skipped_cats.get(cat, 0)
            print("    %-12s %4d seen, %4d dropped (%4.1f%%)"
                  % (cat, n_seen, n_drop, n_drop / max(n_seen, 1) * 100))

    print("saved adapter to", args.out)


if __name__ == "__main__":
    main()
