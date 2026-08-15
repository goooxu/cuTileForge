"""Classifier for sticky CUDA errors. No GPU required."""
from __future__ import print_function

import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from worker import is_cuda_context_error  # noqa: E402


CASES = [
    (True, "AcceleratorError: CUDA error: an illegal memory access was encountered"),
    (True, "RuntimeError: Failed to launch cuTile kernel: an illegal memory access was encountered"),
    (True, "AcceleratorError: CUDA error: an illegal instruction was encountered"),
    (True, "CUDA error: unspecified launch failure"),
    (False, "torch.cuda.OutOfMemoryError: CUDA out of memory"),
    (False, "AssertionError: output mismatch, max diff 1.7"),
    (False, "TileTypeError: int() expects a constant argument"),
    (False, "ValueError: Grid[1] is too big: max=65535, got=65536"),
    (False, "IndentationError: unexpected indent"),
]


def main():
    fails = 0
    for want, msg in CASES:
        got = is_cuda_context_error(msg)
        ok = got is want
        print("  %-4s %s -> %s" % ("ok" if ok else "FAIL", msg[:72], got))
        fails += not ok
    if fails:
        raise SystemExit("%d checks failed" % fails)
    print("all checks passed")


if __name__ == "__main__":
    main()
