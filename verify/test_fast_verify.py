"""Resume helpers for --timing-from. No GPU required."""
from __future__ import print_function

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_verify import passed_untimed, timing_complete  # noqa: E402


def _write(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    return path


def test_passed_untimed():
    prior = {
        "1:0": {"passed": True, "speedup": 1.2},
        "1:1": {"passed": True},
        "2:0": {"passed": False, "stage": "exec"},
    }
    got = passed_untimed(prior)
    ok = got == {"1:1"}
    print("  %-4s passed_untimed %s" % ("ok" if ok else "FAIL", got))
    return 0 if ok else 1


def test_timing_complete():
    fails = 0
    # Hundreds of untimed passes (the GL-C twin hole) is not complete.
    hole = [{"key": "%d:0" % i, "passed": True} for i in range(300)]
    hole += [{"key": "%d:0" % i, "passed": True, "speedup": 1.0}
             for i in range(300, 600)]
    path = _write(hole)
    try:
        got = timing_complete(path)
        ok = got is False
        print("  %-4s twin hole is incomplete" % ("ok" if ok else "FAIL"))
        fails += not ok
    finally:
        os.unlink(path)

    # A handful of leftover timing failures is complete.
    almost = [{"key": "%d:0" % i, "passed": True, "speedup": 1.0}
              for i in range(200)]
    almost += [{"key": "leftover:%d" % i, "passed": True} for i in range(3)]
    path = _write(almost)
    try:
        got = timing_complete(path)
        ok = got is True
        print("  %-4s handful leftover is complete" % ("ok" if ok else "FAIL"))
        fails += not ok
    finally:
        os.unlink(path)

    path = _write([{"key": "0:0", "passed": True, "speedup": 1.0}])
    try:
        got = timing_complete(path, need=10)
        ok = got is False
        print("  %-4s short jsonl is incomplete" % ("ok" if ok else "FAIL"))
        fails += not ok
    finally:
        os.unlink(path)
    return fails


def main():
    fails = test_passed_untimed()
    fails += test_timing_complete()
    if fails:
        raise SystemExit("%d checks failed" % fails)
    print("all checks passed")


if __name__ == "__main__":
    main()
