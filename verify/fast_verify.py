"""Batch correctness-only verifier for generated cuTile kernels.

The benchmark harness (scripts/eval_from_generations.py) runs at roughly 6
samples/min on 4 GPUs, which is far too slow for the tens of thousands of
candidates rejection sampling needs. Measurement showed cuTile's JIT compile is
only ~65 ms, so that cost is dominated by per-sample process startup and the
100 performance trials -- neither of which rejection sampling needs.

The verification itself lives in verify/worker.py, shared with the repair loop
so the pass criterion cannot drift between the two.

Usage:
    python3 verify/fast_verify.py --kernel-dir runs/synth/kernels \\
        --level 90 --out runs/synth/verified.jsonl --workers 16
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import VerifierPool, verify_and_time  # noqa: E402


def load_candidates(kernel_dir: str, level: int, refs: dict, limit=None):
    pattern = re.compile(r"level_%d_problem_(\d+)_sample_(\d+)_kernel\.py" % level)
    tasks = []
    for fname in sorted(os.listdir(kernel_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        pid, sid = int(m.group(1)), int(m.group(2))
        if pid not in refs:
            continue
        with open(os.path.join(kernel_dir, fname)) as f:
            tasks.append(("%d:%d" % (pid, sid), f.read(), refs[pid]))
    return tasks[:limit] if limit else tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-dir", required=True,
                    help="Directory of *_kernel.py files from generate_samples.py.")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--out", required=True, help="JSONL of per-candidate results.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--num-correct-trials", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--measure-time", action="store_true",
                    help="Also time correct candidates against the reference. "
                         "Adds a second, GPU-exclusive phase.")
    ap.add_argument("--num-perf-trials", type=int, default=20)
    ap.add_argument("--ref-mode", default="compile", choices=["compile", "eager"],
                    help="What the reference is timed as. compile is the honest "
                         "comparison: beating eager on a fusion chain mostly "
                         "means beating intermediate materialisation, which "
                         "inductor already removes.")
    args = ap.parse_args()

    from kernelbench.dataset import construct_kernelbench_dataset
    dataset = construct_kernelbench_dataset(args.level)
    refs = {pid: dataset.get_problem_by_id(pid).code
            for pid in dataset.get_problem_ids()}

    tasks = load_candidates(args.kernel_dir, args.level, refs, args.limit)
    print("verifying %d candidates with %d workers over %d GPUs"
          % (len(tasks), args.workers, args.gpus))

    start = time.time()
    if args.measure_time:
        results = verify_and_time(
            tasks, workers=args.workers, gpus=args.gpus,
            num_correct_trials=args.num_correct_trials, timeout_s=args.timeout,
            num_perf_trials=args.num_perf_trials, progress=print,
            ref_mode=args.ref_mode)
    else:
        with VerifierPool(workers=args.workers, gpus=args.gpus,
                          num_correct_trials=args.num_correct_trials,
                          timeout_s=args.timeout) as pool:
            results = pool.verify_batch(tasks)

    n_pass = sum(r["passed"] for r in results.values())
    n_oom = sum(r["stage"] == "oom" for r in results.values())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for key, rec in results.items():
            f.write(json.dumps(rec) + "\n")

    elapsed = time.time() - start
    print("done: %d/%d passed (%.1f%%), %d OOM (inconclusive) in %.1fs -> %.1f cand/s"
          % (n_pass, len(tasks), n_pass / max(len(tasks), 1) * 100,
             n_oom, elapsed, len(tasks) / max(elapsed, 1e-9)))
    if n_oom:
        print("  rerun OOM candidates with fewer --workers to resolve them")

    speedups = sorted(r["speedup"] for r in results.values()
                      if r.get("speedup"))
    if speedups:
        faster = sum(1 for s in speedups if s > 1.0)
        print("  speed: median %.3fx, %d/%d beat the reference"
              % (speedups[len(speedups) // 2], faster, len(speedups)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
