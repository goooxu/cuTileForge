"""Emit synthetic KernelBench-format problems for cuTile training.

The 200 real KernelBench problems are held out as the test set, so every
training task is generated here from the torch API. Output goes to
KernelBench/level{N}/ inside the checkout, which LocalKernelBenchDataset loads
for any positive level number -- so the existing generation, evaluation and
analysis pipeline works on these unchanged.

Usage:
    python3 taskgen/generate_tasks.py --level 90 --tier 2 --count 60
    python3 taskgen/generate_tasks.py --level 91 --curriculum --count 1200
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from operators import BUILDERS, Spec  # noqa: E402

TEMPLATE = '''import torch
import torch.nn as nn
{extra_imports}

class Model(nn.Module):
    """{doc}"""

    def __init__(self{init_sig}):
        super(Model, self).__init__()
{init_body}

    def forward(self, {forward_sig}):
{forward_body}


{consts}

def get_inputs():
{inputs}


def get_init_inputs():
{init_inputs}
'''


def render(spec: Spec) -> str:
    consts = "\n".join("%s = %s" % (k, v) for k, v in spec.consts.items())
    init_sig = (", " + spec.init_sig) if spec.init_sig else ""
    extra = "\n".join(spec.extra_imports)
    return TEMPLATE.format(
        extra_imports=extra,
        doc="%s (tier %d, %s)" % (spec.name, spec.tier, spec.category),
        init_sig=init_sig,
        init_body=spec.init_body,
        forward_sig=spec.forward_sig,
        forward_body=spec.forward_body,
        consts=consts,
        inputs=spec.inputs,
        init_inputs=spec.init_inputs,
    )


def validate(source: str) -> str:
    """Run the problem the way the eval harness will, and return '' if it works.

    A malformed task would show up later as a model failure, so tasks are
    rejected here rather than polluting the pass-rate signal.
    """
    try:
        import torch
        ns = {}
        exec(compile(source, "<task>", "exec"), ns)
        model = ns["Model"](*ns["get_init_inputs"]())
        inputs = ns["get_inputs"]()
        with torch.no_grad():
            out = model(*inputs)
        if not hasattr(out, "shape"):
            return "forward did not return a tensor"
        if out.numel() == 0:
            return "forward returned an empty tensor"
        return ""
    except Exception as e:
        return "%s: %s" % (type(e).__name__, str(e)[:120])


def builder_category(builder, tier: int) -> str:
    """Category a builder emits, discovered by asking it to build one."""
    return builder(tier, random.Random(0)).category


def pick_builders(tier: int, only=None, weight_overrides=None):
    """Builders available at a tier, optionally filtered and reweighted.

    Filtering by category is how a run is pointed at a specific weakness --
    convolution here -- without editing the operator table.
    """
    out = []
    for b, w, tiers in BUILDERS:
        if tier not in tiers:
            continue
        cat = builder_category(b, tier)
        if only and cat not in only:
            continue
        if weight_overrides and cat in weight_overrides:
            w = weight_overrides[cat]
        out.append((b, w))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="kernelbench/KernelBench",
                    help="Directory holding level{N} subdirectories.")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--tier", type=int, default=None,
                    help="Emit a single tier. Mutually exclusive with --curriculum.")
    ap.add_argument("--curriculum", action="store_true",
                    help="Spread across tiers 0-5.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="Empty the level dir first.")
    ap.add_argument("--only-categories", default=None,
                    help="Comma-separated categories to keep, e.g. conv,norm,pool.")
    ap.add_argument("--category-weights", default=None,
                    help="Comma-separated cat=weight overrides, e.g. conv=40,norm=5.")
    args = ap.parse_args()

    if (args.tier is None) == (not args.curriculum):
        ap.error("pass exactly one of --tier or --curriculum")

    rng = random.Random(args.seed)
    out_dir = os.path.join(args.out_root, "level%d" % args.level)
    os.makedirs(out_dir, exist_ok=True)
    if args.clean:
        for f in os.listdir(out_dir):
            if f.endswith(".py"):
                os.remove(os.path.join(out_dir, f))

    only_cats = set(args.only_categories.split(",")) if args.only_categories else None
    weight_overrides = None
    if args.category_weights:
        weight_overrides = {}
        for pair in args.category_weights.split(","):
            cat, _, w = pair.partition("=")
            weight_overrides[cat.strip()] = float(w)

    # Weighted toward the tiers where the model has room to learn: mostly the
    # tiny weak-category tier and the parameterised step above it. Tier 6 is the
    # speed curriculum and is excluded here; ask for it with --tier 6, since
    # mixing it in would put unmeasurably small tasks in a set meant for timing.
    tier_weights = {0: 1, 1: 1, 2: 4, 3: 4, 4: 2, 5: 2}

    written, rejected = 0, 0
    seen_names = set()
    attempts = 0
    while written < args.count and attempts < args.count * 20:
        attempts += 1
        if args.curriculum:
            tiers = list(tier_weights)
            tier = rng.choices(tiers, weights=[tier_weights[t] for t in tiers])[0]
        else:
            tier = args.tier

        builders = pick_builders(tier, only_cats, weight_overrides)
        if not builders:
            continue
        builder = rng.choices([b for b, _ in builders],
                              weights=[w for _, w in builders])[0]

        spec = builder(tier, rng)
        source = render(spec)

        err = validate(source)
        if err:
            rejected += 1
            continue

        pid = written + 1
        fname = "%d_%s_t%d.py" % (pid, spec.name, spec.tier)
        if fname in seen_names:
            continue
        seen_names.add(fname)
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(source)
        written += 1

    print("wrote %d problems to %s (%d candidates rejected by validation)"
          % (written, out_dir, rejected))

    import collections
    tiers = collections.Counter()
    for f in os.listdir(out_dir):
        if not f.endswith(".py"):
            continue
        tiers[f.rsplit("_t", 1)[-1].replace(".py", "")] += 1
    print("by tier:", dict(sorted(tiers.items())))


if __name__ == "__main__":
    main()
