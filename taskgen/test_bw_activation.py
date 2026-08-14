#!/usr/bin/env python3
"""The bandwidth-bound activation builder must not recreate the last speed miss.

It has to emit the KernelBench activation *regime* (around a billion elements)
without emitting the KernelBench activation *shape*, and it must not sneak
softmax back in under the activation label.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from operators import (  # noqa: E402
    ACTIVATION_OPS, BUILDERS, BW_2D, POINTWISE_ACTIVATION_OPS, _KB_ACT_SHAPE,
    large_pointwise_activation, long_elementwise_chain,
)
from generate_tasks import _is_huge_pointwise  # noqa: E402


def check(cond, msg):
    print("  %-4s %s" % ("ok" if cond else "FAIL", msg))
    return 0 if cond else 1


def main() -> None:
    fails = 0
    fails += check(_KB_ACT_SHAPE not in BW_2D,
                   "BW_2D does not contain the KernelBench test shape")
    fails += check(all(0.8e9 <= m * k <= 1.4e9 for m, k in BW_2D),
                   "every BW_2D shape is 0.8e9-1.4e9 elements")
    fails += check(all(op[0] not in {"Softmax", "LogSoftmax", "Softmin"}
                       for op in POINTWISE_ACTIVATION_OPS),
                   "POINTWISE_ACTIVATION_OPS excludes the softmax family")
    fails += check(len(POINTWISE_ACTIVATION_OPS) == len(ACTIVATION_OPS) - 3,
                   "exactly the three softmax-family ops were dropped")

    rng = random.Random(0)
    seen_ops, seen_shapes = set(), set()
    bad_shape = bad_test = bad_cat = bad_sm = 0
    for _ in range(400):
        spec = large_pointwise_activation(6, rng)
        seen_ops.add(spec.name)
        shape = (spec.consts["batch_size"], spec.consts["dim"])
        seen_shapes.add(shape)
        if spec.category != "activation":
            bad_cat += 1
        if shape == _KB_ACT_SHAPE:
            bad_test += 1
        if shape not in BW_2D:
            bad_shape += 1
        if "softmax" in spec.forward_body.lower():
            bad_sm += 1
    fails += check(bad_cat == 0, "every spec is category=activation")
    fails += check(bad_test == 0, "no emitted shape is the KernelBench test shape")
    fails += check(bad_shape == 0, "every emitted shape is on the BW_2D ladder")
    fails += check(bad_sm == 0, "no forward is a softmax")
    fails += check(seen_ops <= {n for n, _ in POINTWISE_ACTIVATION_OPS},
                   "only pointwise activation labels appear")
    fails += check(len(seen_shapes) >= 8,
                   "sampling covers most of the shape ladder (%d distinct)"
                   % len(seen_shapes))
    names = [b[0].__name__ for b in BUILDERS]
    fails += check("large_pointwise_activation" in names
                   and "long_elementwise_chain" in names,
                   "both builders stay registered; the insert must not eat the chain")
    fails += check(callable(long_elementwise_chain),
                   "long_elementwise_chain is still a function")
    fails += check(_is_huge_pointwise({"batch_size": 4096, "dim": 327680}),
                   "BW_2D activations skip the eager forward")
    fails += check(not _is_huge_pointwise({"batch_size": 8192, "dim": 8192}),
                   "PERF_2D 8192x8192 still runs a real forward")
    fails += check(not _is_huge_pointwise(
        {"batch_size": 32, "in_channels": 64, "out_channels": 128,
         "height": 256, "width": 256, "kernel_size": 3}),
                   "a large conv is not skipped just because consts multiply big")
    print("\n%s" % ("all checks passed" if not fails else "%d FAILED" % fails))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
