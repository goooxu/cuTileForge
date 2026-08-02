"""LoRA target modules for Qwen3-Next.

Qwen3-Next interleaves Gated DeltaNet (linear attention) with full attention at
full_attention_interval=4, so of this model's 48 layers only 12 carry
self_attn.{q,k,v,o}_proj. Targeting just those standard names -- the default
almost everywhere -- reaches a quarter of the network, yields a trainable
fraction near 0.02%, and is widely reported to produce NaN loss. The DeltaNet
projections have to be included.

The 512 routed experts hold most of the parameters, but attaching adapters to
each of them across 48 layers is not workable through peft. The working
assumption here is that attention, DeltaNet and the shared expert carry enough
capacity for what this run is teaching, which is a set of rules -- grids are at
most 3D, tile rank must match array rank, an Array is not a tensor -- rather
than new domain knowledge. resolve_targets() reports the trainable fraction so
that assumption is visible rather than silent.
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

# Shared expert, present in every MoE layer. The routed experts are left alone.
SHARED_EXPERT_TARGETS = [
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_TARGETS = DELTANET_TARGETS + ATTENTION_TARGETS + SHARED_EXPERT_TARGETS


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
