"""Per-architecture training knobs: which class loads a base model, what to adapt.

Two families run through this trainer and they share almost nothing below the
tokenizer, so the differences live here instead of in branches scattered
through train_lora.py, smoke_lora.py, probe_trainable.py and merge_lora.py.

Qwen3-Next interleaves Gated DeltaNet (linear attention) with full attention at
full_attention_interval=4, so of this model's 48 layers only 12 carry
self_attn.{q,k,v,o}_proj. Targeting just those standard names -- the default
almost everywhere -- reaches a quarter of the network, yields a trainable
fraction near 0.02%, and is widely reported to produce NaN loss. The DeltaNet
projections have to be included.

Muse Glimmer is the easy case by comparison: 52 dense layers of SwiGLU with
gated GQA, no linear attention and no fused expert tensors, so LoRA there is
real LoRA. Its one trap is the 1.8B perception encoder, whose 50 layers repeat
the names q_proj/k_proj/v_proj -- see MUSE_ATTENTION_TARGETS.

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
and why every checkpoint write takes minutes.

**And it was never necessary.** Trained on identical data, ATTENTION_ONLY_TARGETS
at r=128 -- 137M trainable, 0.17% -- matches it: 29.2% against 29.8% pass@1 and 45
against 46 problems solved on Level 1. It also trains in a third of the time, at
46 GB instead of 84 GB, and saves a 537 MB adapter instead of 26 GB. Prefer it.
DEFAULT_TARGETS is kept because every result before phase 8 was produced with it
and removing it would make those numbers unreproducible.

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

QWEN3_NEXT = "qwen3_next"
MUSE_GLIMMER = "muse_glimmer"

# Muse Glimmer targets are anchored paths, not leaf names, and that is not
# stylistic: the perception encoder's 50 layers carry q_proj / k_proj / v_proj
# too, so a bare name list would adapt a vision tower this task never uses.
# peft treats a string target_modules as a regex it fullmatches against each
# module's name, which is what lets the path do the excluding.
#
# self_attn.gate_proj is deliberate rather than a typo for the MLP's: this model
# has gated attention, so gate_proj appears 104 times on the language side --
# once per layer in self_attn and once in mlp.
_MUSE_LAYER = r"model\.language_model\.layers\.\d+"
MUSE_ATTENTION_TARGETS = (
    _MUSE_LAYER + r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj)")
MUSE_DEFAULT_TARGETS = (
    _MUSE_LAYER + r"\.(self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj)"
                  r"|mlp\.(gate_proj|up_proj|down_proj))")

# Substrings that place a module in the perception encoder rather than the
# language model. Used to assert a text-only run left the vision side alone.
VISION_MARKERS = ("vision", "visual", "perception")


def family_of(model_or_config) -> str:
    """The `model_type` string, from a model or a bare config."""
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "model_type", "") or ""


def load_base_model(path, device_map="auto", dtype=None, trust_remote_code=True,
                    max_memory=None):
    """Load a base model with the class its own config names.

    AutoModelForCausalLM does not cover every family here. Muse Glimmer is
    registered only under the multimodal mapping, so the causal-LM mapping
    returns None and from_pretrained raises before reading a single shard.
    Resolving the class from config.architectures picks the right one for both
    and keeps callers from having to know which.
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

    if dtype is None:
        dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    family = family_of(config)
    if MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(family):
        cls = AutoModelForCausalLM
    else:
        arch = (getattr(config, "architectures", None) or [None])[0]
        cls = getattr(transformers, arch, None) if arch else None
        if cls is None:
            raise SystemExit(
                "cannot load %s: model_type %r is not a causal LM and "
                "transformers has no class named %r"
                % (path, family, arch))
    print("loading %s as %s (model_type %s)" % (path, cls.__name__, family))
    kwargs = dict(dtype=dtype, device_map=device_map,
                  trust_remote_code=trust_remote_code)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    return cls.from_pretrained(path, **kwargs)


def logit_transform(model_or_config):
    """The model's own post-LM-head transform, or None if it does not have one.

    This trainer computes cross-entropy from the LM head itself to skip the head
    on unlabelled positions, so anything the model's forward does between the
    head and its loss has to be repeated here or the number is simply wrong.
    Muse Glimmer pre-scales by output_multiplier and then Gemma-style
    tanh-softcaps at final_logit_softcapping; leaving that out put the loss at
    54.8 where the model's own labels= path reported 13.3.

    Applied in the head's dtype rather than after the fp32 upcast, matching the
    order in modeling_muse_glimmer so the two paths agree to bf16 precision.
    """
    import torch

    config = getattr(model_or_config, "config", model_or_config)
    inner = getattr(config, "text_config", None) or config
    cap = getattr(inner, "final_logit_softcapping", None)
    mult = getattr(inner, "output_multiplier", None)
    if not cap and not mult:
        return None

    def apply(logits):
        if not logits.is_contiguous():
            logits = logits.contiguous()
        if mult:
            logits.mul_(mult)
        if cap:
            logits.div_(cap)
            logits.tanh_()
            logits.mul_(cap)
        return logits

    return apply


def targets_for(model_or_config, choice="default"):
    """Target modules for this family, as a name list or an anchored regex."""
    family = family_of(model_or_config)
    if family == MUSE_GLIMMER:
        return (MUSE_ATTENTION_TARGETS if choice == "attention_only"
                else MUSE_DEFAULT_TARGETS)
    return (ATTENTION_ONLY_TARGETS if choice == "attention_only"
            else DEFAULT_TARGETS)


def matched_modules(model, targets):
    """Names of the nn.Linear modules a peft target spec would wrap.

    Mirrors peft's own matching: a string is fullmatched as a regex against the
    module name, a list matches on the trailing name component.
    """
    import re

    import torch.nn as nn

    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(targets, str):
            if re.fullmatch(targets, name):
                names.append(name)
        elif name.rsplit(".", 1)[-1] in targets:
            names.append(name)
    return names


def validate_targets(model, targets, verbose=True):
    """Refuse to train on a spec that silently adapts almost nothing.

    peft does not complain when a target name reaches one module out of a
    thousand, and the run then spends hours optimising 0.02% of the weights.
    Each family has its own version of that failure, so each gets checked:
    Qwen3-Next must reach the DeltaNet projections or only a quarter of the
    layers are adapted at all, and Muse Glimmer must not reach the vision
    tower, which for a text-only task is capacity spent on nothing.
    """
    import torch.nn as nn

    family = family_of(model)
    names = matched_modules(model, targets)
    if verbose:
        counts = {}
        for name in names:
            counts[name.rsplit(".", 1)[-1]] = counts.get(
                name.rsplit(".", 1)[-1], 0) + 1
        total = sum(m.weight.numel() for m in model.modules()
                    if isinstance(m, nn.Linear))
        matched = sum(m.weight.numel() for n, m in model.named_modules()
                      if isinstance(m, nn.Linear) and n in set(names))
        print("LoRA target match (%s):" % (family or "?"))
        for leaf in sorted(counts):
            print("  %-14s %5d modules" % (leaf, counts[leaf]))
        print("  %d modules, %.2f%% of nn.Linear parameters"
              % (len(names), matched / total * 100 if total else 0.0))

    if not names:
        raise SystemExit("target spec %r matched no nn.Linear module" % (targets,))

    if isinstance(targets, list):
        counts = {t: 0 for t in targets}
        for name in names:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in counts:
                counts[leaf] += 1
        missing = [t for t, c in counts.items() if c == 0]
        if missing:
            raise SystemExit("target modules match nothing: %s"
                             % ", ".join(missing))
        if family == QWEN3_NEXT and not any(counts[t] for t in DELTANET_TARGETS):
            raise SystemExit("no DeltaNet projections matched")

    vision = [n for n in names
              if any(marker in n for marker in VISION_MARKERS)]
    if vision:
        raise SystemExit(
            "%d target modules are in the vision tower (e.g. %s); this is a "
            "text-only task" % (len(vision), vision[0]))
    return names


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
