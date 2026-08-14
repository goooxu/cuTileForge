#!/usr/bin/env python3
"""The solid-mode speed band has to reject both dead zones."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_frontier import in_speed_band  # noqa: E402


def check(cond, msg):
    print("  %-4s %s" % ("ok" if cond else "FAIL", msg))
    return 0 if cond else 1


def main() -> None:
    fails = 0
    fails += check(in_speed_band(0.55, 0.25, 1.0),
                   "0.55x, the KernelBench activation regime, is in the band")
    fails += check(not in_speed_band(0.10, 0.25, 1.0),
                   "0.10x is inside the clamp and is dropped")
    fails += check(not in_speed_band(0.25, 0.25, 1.0),
                   "the lower bound is exclusive")
    fails += check(not in_speed_band(1.0, 0.25, 1.0),
                   "already matching the reference is dropped")
    fails += check(not in_speed_band(6.0, 0.25, 1.0),
                   "already-winning 6x tasks are dropped")
    fails += check(not in_speed_band(None, 0.25, 1.0),
                   "a missing timing is never in the band")
    fails += check(in_speed_band(0.10, None, 1.0),
                   "with no min bound, slow tasks still pass the max filter")
    fails += check(in_speed_band(2.0, 0.25, None),
                   "with no max bound, fast tasks still pass the min filter")
    print("\n%s" % ("all checks passed" if not fails else "%d FAILED" % fails))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
