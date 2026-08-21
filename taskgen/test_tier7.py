#!/usr/bin/env python3
"""Tier 7 must only emit catchable families at CATCH_2D sizes."""
import random
import sys

from operators import CATCH_2D  # noqa: E402
from generate_tasks import pick_builders as _pick  # noqa: E402

ALLOWED = {"activation", "elementwise", "norm"}
BLOCK = {"matmul", "conv"}


def main():
    builders = _pick(7)
    names = sorted({b.__name__ for b, _ in builders})
    print("tier 7 builders:", names)
    if any("matmul" in n or "conv" in n for n in names):
        raise SystemExit("tier 7 listed a matmul/conv builder: %s" % names)
    rng = random.Random(0)
    for i in range(40):
        b = rng.choices([x for x, _ in builders],
                        weights=[w for _, w in builders])[0]
        spec = b(7, rng)
        if spec.category in BLOCK:
            raise SystemExit("%s emitted %s" % (b.__name__, spec.category))
        if spec.category not in ALLOWED:
            raise SystemExit("%s emitted unexpected %s" % (b.__name__, spec.category))
        n = spec.consts.get("batch_size", spec.consts.get("M"))
        k = spec.consts.get("dim", spec.consts.get("K"))
        if n is None or k is None:
            raise SystemExit("%s missing 2d size: %s" % (b.__name__, spec.consts))
        if (int(n), int(k)) not in CATCH_2D:
            raise SystemExit("%s shape %s x %s not in CATCH_2D" %
                             (b.__name__, n, k))
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
