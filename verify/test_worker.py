"""Classifier for sticky CUDA errors. No GPU required."""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import (  # noqa: E402
    is_compiler_timeout,
    is_cuda_context_error,
    kill_stray_compilers,
)


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


class _TileCompilerTimeoutError(Exception):
    pass


def test_compiler_timeout_classifier():
    fails = 0
    cases = [
        (True, _TileCompilerTimeoutError(
            "`tileiras` compiler exceeded timeout 180s. "
            "Using a smaller tile size may reduce compilation time.")),
        (True, "TileCompilerTimeoutError: tileiras compiler exceeded timeout 10s"),
        (False, "AssertionError: output mismatch, max diff 1.7"),
        (False, "torch.cuda.OutOfMemoryError: CUDA out of memory"),
    ]
    # Rename the dummy so is_compiler_timeout sees TileCompilerTimeoutError.
    _TileCompilerTimeoutError.__name__ = "TileCompilerTimeoutError"
    for want, msg in cases:
        got = is_compiler_timeout(msg)
        ok = got is want
        print("  %-4s compiler_timeout %s -> %s"
              % ("ok" if ok else "FAIL", type(msg).__name__, got))
        fails += not ok
    return fails


def test_kill_stray_compilers():
    """A fake compiler whose argv contains the temp dir is killed; others are not."""
    import subprocess
    import sys
    import tempfile
    import time

    d = tempfile.mkdtemp(prefix="cutile-killtest-")
    py = sys.executable
    other = subprocess.Popen(
        [py, "-c", "import time; time.sleep(30)  # ptxas unrelated"])
    ours = subprocess.Popen(
        [py, "-c", "import time; time.sleep(30)  # ptxas %s" % d])
    time.sleep(0.3)
    try:
        # root_pid=-1: no descendants, so only the temp-dir marker may match.
        killed = kill_stray_compilers(d, root_pid=-1)
        time.sleep(0.2)
        ours.poll()
        other.poll()
        ours_dead = ours.returncode is not None
        other_alive = other.returncode is None
        print("  killed=%s ours_dead=%s other_alive=%s"
              % (killed, ours_dead, other_alive))
        if not ours_dead:
            print("  FAIL: compiler with temp dir still alive")
            return 1
        if not other_alive:
            print("  FAIL: unrelated ptxas was killed")
            return 1
        print("  ok   kill_stray_compilers")
        return 0
    finally:
        for p in (ours, other):
            if p.poll() is None:
                p.kill()
                p.wait()


def main():
    fails = 0
    for want, msg in CASES:
        got = is_cuda_context_error(msg)
        ok = got is want
        print("  %-4s %s -> %s" % ("ok" if ok else "FAIL", msg[:72], got))
        fails += not ok
    fails += test_compiler_timeout_classifier()
    fails += test_kill_stray_compilers()
    if fails:
        raise SystemExit("%d checks failed" % fails)
    print("all checks passed")


if __name__ == "__main__":
    main()
