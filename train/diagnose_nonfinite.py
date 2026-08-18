"""Find which part of the training setup turns a long sequence non-finite.

Roughly a third of this dataset's ~16k-token examples produce a non-finite loss
during training, while the same examples are perfectly finite under eval mode
with no_grad. So the trigger is something the trainer does rather than the data,
and the trainer can only observe the end result.

This reuses the trainer's own dataset and collate so the token stream and label
masking are identical, then toggles one setting at a time -- train mode, LoRA,
gradient checkpointing, and its reentrant variant -- to name the one responsible.

Usage:
    python3 train/diagnose_nonfinite.py --model /raid/... --data runs/sft_A_full.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3, help="Longest examples to test.")
    ap.add_argument("--max-len", type=int, default=20480)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model

    from lora_config import load_base_model, targets_for
    from train_lora import CompletionOnlyDataset, collate

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = CompletionOnlyDataset(args.data, tok, args.max_len)

    order = sorted(range(len(ds)), key=lambda i: -len(ds[i]["input_ids"]))[:args.n]
    print("testing token lengths:", [len(ds[i]["input_ids"]) for i in order])

    model = load_base_model(args.model, device_map="auto", dtype=torch.bfloat16)
    dev = next(model.parameters()).device
    targets = targets_for(model)

    def run(tag, use_lora, train_mode, ckpt, reentrant=None):
        m = model
        if use_lora:
            m = get_peft_model(model, LoraConfig(
                r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                task_type="CAUSAL_LM", target_modules=targets))
        if ckpt:
            if reentrant is None:
                m.gradient_checkpointing_enable()
            else:
                m.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": reentrant})
            if hasattr(m, "enable_input_require_grads"):
                m.enable_input_require_grads()
        m.train() if train_mode else m.eval()
        m.config.use_cache = False

        results = []
        for i in order:
            input_ids, labels, attn, _cats = collate([ds[i]], pad_id)
            out = m(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
                    labels=labels.to(dev))
            results.append("%.4f" % out.loss.item()
                           if torch.isfinite(out.loss) else "NaN/Inf")
            del out
            torch.cuda.empty_cache()
        print("  %-46s %s" % (tag, "  ".join(results)))

        if ckpt:
            m.gradient_checkpointing_disable()
        if use_lora and hasattr(m, "unload"):
            m.unload()
        torch.cuda.empty_cache()

    def run_mlp_only(tag):
        """Checkpoint only the MoE MLPs, leaving the stateful layers alone.

        Qwen3-Next interleaves Gated DeltaNet layers, which carry a recurrent
        state, with MoE MLPs, which are pure functions of their input. Whole-layer
        checkpointing is what breaks, so this tests whether confining it to the
        part that is safe to recompute keeps the loss finite while still saving
        most of the activation memory.
        """
        from torch.utils.checkpoint import checkpoint

        model.gradient_checkpointing_disable()
        model.train()
        model.config.use_cache = False

        patched = []
        for layer in model.model.layers:
            mlp = layer.mlp
            orig = mlp.forward

            def wrapped(*a, _f=orig, **kw):
                return checkpoint(_f, *a, use_reentrant=False, **kw)

            mlp.forward = wrapped
            patched.append((mlp, orig))

        model.enable_input_require_grads()
        results = []
        for i in order:
            input_ids, labels, attn, _cats = collate([ds[i]], pad_id)
            out = model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
                        labels=labels.to(dev))
            results.append("%.4f" % out.loss.item()
                           if torch.isfinite(out.loss) else "NaN/Inf")
            del out
            torch.cuda.empty_cache()
        print("  %-46s %s" % (tag, "  ".join(results)))
        print("  %-46s peak %.0f GB"
              % ("", max(torch.cuda.max_memory_allocated(i)
                         for i in range(torch.cuda.device_count())) / 1e9))

        for mlp, orig in patched:
            mlp.forward = orig

    print("\nloss per example under each configuration:\n")
    with torch.no_grad():
        run("eval, no grad, no lora, no ckpt", False, False, False)
    run("train mode, no lora, no ckpt", False, True, False)
    print("  %-46s peak %.0f GB"
          % ("", max(torch.cuda.max_memory_allocated(i)
                     for i in range(torch.cuda.device_count())) / 1e9))
    run("train mode, no lora, ckpt (default)", False, True, True)
    run("train mode, no lora, ckpt reentrant=False", False, True, True, False)
    run_mlp_only("train mode, no lora, ckpt on MoE MLPs only")


if __name__ == "__main__":
    main()
