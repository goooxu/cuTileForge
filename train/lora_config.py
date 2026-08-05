"""LoRA target modules for Qwen3-Next.

Qwen3-Next interleaves Gated DeltaNet (linear attention) with full attention at
full_attention_interval=4, so of this model's 48 layers only 12 carry
self_attn.{q,k,v,o}_proj. Targeting just those standard names -- the default
almost everywhere -- reaches a quarter of the network, yields a trainable
fraction near 0.02%, and is widely reported to produce NaN loss. The DeltaNet
projections have to be included.

A warning about what this config actually does, measured rather than assumed
(train/probe_trainable.py prints the breakdown):

    experts base_layer   4.83B  trainable -- FULL fine-tune, not LoRA
    experts lora_A/B     2.01B  trainable
    attention + DeltaNet 0.03B  trainable
                         6.88B  total, 7.9% of the model

gate_proj/up_proj/down_proj were meant to reach only the shared expert. They also
match the routed experts' fused tensors, which are not nn.Linear, so peft wraps
them with lora.ParamWrapper -- and that leaves the original weight trainable.
So 99.5% of what this trains is the routed experts, and 70% of that is a full
fine-tune of weights the name "LoRA" implies are frozen.

That is why the adapter is 27.5 GB rather than the ~70 MB an attention-only LoRA
would produce, why memory has been tight enough to force gradient checkpointing,
and why every checkpoint write takes minutes. It has produced the results so far,
so it is left alone for supervised rounds, but anything that checkpoints often --
reinforcement learning above all -- should freeze the expert parameters and train
only the attention and DeltaNet adapters.

resolve_targets() reports the nn.Linear coverage, which is what it was written
for; note it counts only nn.Linear and so cannot see the fused expert tensors
that dominate the real total.
"""

# Gated DeltaNet layers (36 of 48 here). Names verified against this model
# rather than copied from write-ups: transformers' Qwen3Next fuses the linear
# attention inputs into in_proj_qkvz (q, k, v, z) and in_proj_ba (beta, alpha).
# Community configs for Qwen3.5/3.6 list in_proj_qkv and in_proj_z, which match
# nothing here and would silently leave the DeltaNet input side unadapted.
DELTANET_TARGETS = [
    "in_proj_qkvz",
    "in_proj_ba",
    "out_proj",
]

# Full-attention layers (12 of 48 here).
ATTENTION_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# Intended as the shared expert, but these names also match the routed experts'
# fused tensors; see the module docstring for what that actually costs.
SHARED_EXPERT_TARGETS = [
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_TARGETS = DELTANET_TARGETS + ATTENTION_TARGETS + SHARED_EXPERT_TARGETS

# For runs that cannot afford 27.5 GB per checkpoint. Reinforcement learning
# sharpens a distribution the model already has rather than teaching it new
# material, so it needs far less capacity than supervised training does.
ATTENTION_ONLY_TARGETS = DELTANET_TARGETS + ATTENTION_TARGETS


def freeze_experts(model) -> tuple[int, int]:
    """Leave only the attention and DeltaNet adapters trainable.

    Applied after loading an adapter that was trained with the full target set,
    so the expert weights it learned are kept and simply stop receiving
    gradients. Returns (trainable, frozen) parameter counts.
    """
    trainable = frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad and "expert" in name.lower():
            p.requires_grad = False
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return trainable, frozen


def resolve_targets(model, targets=None, verbose=True):
    """Report which target names actually match modules, and how much they cover.

    A silent near-zero match is the failure mode this guards against, so the
    caller gets a per-name count and can refuse to train on a bad config.
    """
    import torch.nn as nn

    targets = list(targets or DEFAULT_TARGETS)
    counts = {t: 0 for t in targets}
    matched_params = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n_param = module.weight.numel()
            total_params += n_param
            leaf = name.rsplit(".", 1)[-1]
            if leaf in counts:
                counts[leaf] += 1
                matched_params += n_param

    if verbose:
        print("LoRA target match:")
        for t in targets:
            flag = "" if counts[t] else "   <-- MATCHES NOTHING"
            print("  %-14s %5d modules%s" % (t, counts[t], flag))
        pct = matched_params / total_params * 100 if total_params else 0.0
        print("  covers %.2f%% of nn.Linear parameters" % pct)

    return counts, matched_params, total_params
