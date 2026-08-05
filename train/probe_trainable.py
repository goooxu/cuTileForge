"""Report which parameters a LoRA config actually makes trainable, by group.

train_lora.py prints one aggregate number, 6.88B trainable, which turns out to
hide something important: 99.5% of the saved adapter sits on the routed experts,
and 4.83B of that is under `base_layer`, the name peft gives the original frozen
weight. Either those are frozen and are being written to a 27.5 GB checkpoint for
nothing, or they are trainable and this is not LoRA on the experts at all but a
full fine-tune of them. The two have very different costs, so measure it.

Usage:
    python3 train/probe_trainable.py --model /raid/tmp/.../Qwen3-Coder-Next
"""

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def group(name: str) -> str:
    m = re.search(r"layers\.\d+\.(.*)", name)
    leaf = m.group(1) if m else name
    leaf = re.sub(r"^(mlp|self_attn|linear_attn)\.", "", leaf)
    # Collapse the lora_A/lora_B/base_layer suffix into the part that matters.
    for tag in ("base_layer", "lora_A", "lora_B", "lora_embedding"):
        if tag in leaf:
            head = leaf.split("." + tag)[0]
            return "%s [%s]" % (head, tag)
    return leaf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-r", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    from lora_config import DEFAULT_TARGETS

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=DEFAULT_TARGETS))

    train = collections.Counter()
    frozen = collections.Counter()
    for name, p in model.named_parameters():
        (train if p.requires_grad else frozen)[group(name)] += p.numel()

    print("\n%-46s %16s" % ("trainable group", "params"))
    for g, n in train.most_common():
        print("%-46s %16s" % (g[:46], "{:,}".format(n)))
    print("%-46s %16s" % ("TOTAL TRAINABLE", "{:,}".format(sum(train.values()))))

    base_trainable = sum(n for g, n in train.items() if "base_layer" in g)
    print("\ntrainable and named base_layer: %s" % "{:,}".format(base_trainable))
    print("-> %s" % ("this is a full fine-tune of those weights, not LoRA"
                     if base_trainable else
                     "base weights are frozen; saving them is pure checkpoint bloat"))


if __name__ == "__main__":
    main()
