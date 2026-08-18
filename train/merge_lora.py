"""Merge a LoRA adapter into the base weights and save a servable model.

vLLM's LoRA support does not cover every module type this adapter touches on a
Qwen3-Next MoE, and the evaluation has to run through the same vLLM path as the
baseline for the comparison to mean anything. Merging sidesteps both issues at
the cost of a second full copy of the weights on disk.

Usage:
    python3 train/merge_lora.py --base /ws/models/Qwen3-Coder-Next \\
        --adapter /ws/models/lora-cutile --out /ws/models/Qwen3-Coder-Next-cutile
"""

import argparse
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_config import load_base_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from peft import PeftModel

    print("loading base ...")
    t0 = time.time()
    model = load_base_model(args.base, device_map="cpu", dtype=torch.bfloat16)
    print("  %.1fs" % (time.time() - t0))

    print("applying adapter ...")
    model = PeftModel.from_pretrained(model, args.adapter, device_map="cpu")

    print("merging ...")
    t0 = time.time()
    model = model.merge_and_unload()
    print("  %.1fs" % (time.time() - t0))

    os.makedirs(args.out, exist_ok=True)
    print("saving to %s ..." % args.out)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(args.out)

    # vLLM reads the chat template from here; without it the served model would
    # be prompted differently from the baseline. Multimodal checkpoints also need
    # their processor config, or AutoProcessor cannot construct and the server
    # refuses to start on weights that are otherwise fine.
    for extra in ("chat_template.jinja", "generation_config.json",
                  "processor_config.json", "preprocessor_config.json"):
        src = os.path.join(args.base, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, extra))
            print("  copied %s" % extra)

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print("done: %.1f GB in %s" % (total / 1e9, args.out))


if __name__ == "__main__":
    main()
