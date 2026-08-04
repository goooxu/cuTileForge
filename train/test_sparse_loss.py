"""Check the sparse loss path against the model's own labels= computation.

sparse_lm_loss skips the LM head for unlabelled positions to save memory. That
is only sound if it returns the same number as passing labels= to the model, so
this compares them on real examples before a training run depends on it.

Usage:
    python3 train/test_sparse_loss.py --model /raid/... --data runs/sft_A_full.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train_lora import CompletionOnlyDataset, collate, sparse_lm_loss

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # Short sequences only: the reference path has to build full logits, which is
    # the very cost this is avoiding.
    ds = CompletionOnlyDataset(args.data, tok, 4096)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    model.config.use_cache = False
    dev = next(model.parameters()).device

    def compare(tag, m):
        worst_local = 0.0
        print("\n%s" % tag)
        print("%-8s %14s %14s %12s"
              % ("example", "labels= loss", "sparse loss", "abs diff"))
        with torch.no_grad():
            for i in range(min(args.n, len(ds))):
                input_ids, labels, attn = collate([ds[i]], pad_id)
                input_ids = input_ids.to(dev)
                labels, attn = labels.to(dev), attn.to(dev)

                ref = m(input_ids=input_ids, attention_mask=attn, labels=labels).loss
                got = sparse_lm_loss(m, input_ids, attn, labels)
                d = abs(ref.item() - got.item())
                worst_local = max(worst_local, d)
                print("%-8d %14.6f %14.6f %12.2e" % (i, ref.item(), got.item(), d))
        return worst_local

    worst = compare("plain causal LM", model)

    # The training path wraps the model in PEFT, whose attribute forwarding is
    # exactly what unwrap() has to see through, so cover that too.
    from peft import LoraConfig, get_peft_model
    from train_lora import DEFAULT_TARGETS

    peft_model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=DEFAULT_TARGETS))
    peft_model.eval()
    worst = max(worst, compare("PEFT-wrapped", peft_model))

    # bf16 activations make exact equality unrealistic; this is the scale of
    # difference expected from reassociating the same reduction.
    print("\nworst difference %.2e -> %s" % (worst, "OK" if worst < 2e-3 else "MISMATCH"))
    raise SystemExit(0 if worst < 2e-3 else 1)


if __name__ == "__main__":
    main()
