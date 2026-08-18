"""Gate G3: confirm a base model can actually be LoRA fine-tuned on this hardware.

Checks the three things that would sink the training stage:
  1. the backward pass works through whatever this architecture does instead of
     plain attention (Qwen3-Next's Gated DeltaNet has a pure-PyTorch fallback,
     so autograd should cover it even without fused kernels)
  2. the LoRA target modules match a meaningful share of the network rather than
     the ~0.02% that the standard attention-only names would reach, and on a
     multimodal checkpoint that they stay out of the vision tower
  3. a handful of optimiser steps reduce the loss without producing NaN

Run inside the training image:
    python3 train/smoke_lora.py --model /ws/models/Qwen3-Coder-Next
    python3 train/smoke_lora.py --model /raid/tmp/.../Muse-Glimmer-30B
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_config import (family_of, load_base_model,  # noqa: E402
                         targets_for, validate_targets)


def n_layers(cfg):
    """Layer count, wherever this config keeps it.

    Multimodal configs nest the decoder's own config, so the attribute is not on
    the top level: Muse Glimmer has text_config.num_hidden_layers.
    """
    for holder in (cfg, getattr(cfg, "text_config", None)):
        if holder is not None and getattr(holder, "num_hidden_layers", None):
            return holder.num_hidden_layers
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--targets", default="default",
                    choices=["default", "attention_only"])
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("torch", torch.__version__, "| GPUs", torch.cuda.device_count())
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print("arch", cfg.architectures, "| model_type", family_of(cfg),
          "| layers", n_layers(cfg))
    if getattr(cfg, "full_attention_interval", None):
        print("full_attention_interval", cfg.full_attention_interval)

    import importlib.util
    print("flash-linear-attention: %s"
          % ("present (fused path)" if importlib.util.find_spec("fla")
             else "absent (torch fallback, differentiable)"))

    t0 = time.time()
    print("loading weights ...")
    model = load_base_model(args.model, device_map="auto", dtype=torch.bfloat16)
    print("loaded in %.1fs" % (time.time() - t0))
    print("device map spans:", sorted({str(p.device) for p in model.parameters()}))

    targets = targets_for(model, args.targets)
    validate_targets(model, targets)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    allp = sum(p.numel() for p in model.parameters())
    print("\ntrainable %s / %s (%.4f%%)"
          % (f"{trainable:,}", f"{allp:,}", trainable / allp * 100))
    if trainable / allp * 100 < 0.01:
        print("FAIL: trainable fraction is in the range that reportedly NaNs")
        return

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = len(tok)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)

    torch.manual_seed(0)
    first_dev = next(model.parameters()).device
    losses = []
    # A fixed random batch: the point is that the loss on it goes down, which
    # only happens if gradients actually reach the adapters.
    ids = torch.randint(0, vocab, (1, args.seq_len), device=first_dev)

    print("\nstep  loss")
    for step in range(args.steps):
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        if not torch.isfinite(loss):
            print("FAIL: loss is %s at step %d" % (loss.item(), step))
            return
        loss.backward()

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        losses.append(loss.item())
        print("%4d  %.4f   (grad norm %.3f)" % (step, loss.item(), gnorm))

    print()
    print("peak GPU memory: %.1f GB"
          % (max(torch.cuda.max_memory_allocated(i)
                 for i in range(torch.cuda.device_count())) / 1e9))
    if losses[-1] < losses[0]:
        print("PASS: loss fell %.4f -> %.4f, no NaN" % (losses[0], losses[-1]))
    else:
        print("WARN: loss did not fall (%.4f -> %.4f); check lr"
              % (losses[0], losses[-1]))


if __name__ == "__main__":
    main()
