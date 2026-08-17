"""Batch correctness-only verifier for generated cuTile kernels.

The benchmark harness (scripts/eval_from_generations.py) runs at roughly 6
samples/min on 4 GPUs, which is far too slow for the tens of thousands of
candidates rejection sampling needs. Measurement showed cuTile's JIT compile is
only ~65 ms, so that cost is dominated by per-sample process startup and the
100 performance trials -- neither of which rejection sampling needs.

The verification itself lives in verify/worker.py, shared with the repair loop
so the pass criterion cannot drift between the two. `--timeout` covers tileiras
and ptxas: overtime is a `timeout` failure, not a hung worker.

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
from worker import INCONCLUSIVE_STAGES, VerifierPool, verify_and_time  # noqa: E402


def load_candidates(kernel_dir: str, level: int, refs: dict, limit=None,
                    only_keys=None):
    pattern = re.compile(r"level_%d_problem_(\d+)_sample_(\d+)_kernel\.py" % level)
    tasks = []
    for fname in sorted(os.listdir(kernel_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        pid, sid = int(m.group(1)), int(m.group(2))
        if pid not in refs:
            continue
        key = "%d:%d" % (pid, sid)
        if only_keys is not None and key not in only_keys:
            continue
        with open(os.path.join(kernel_dir, fname)) as f:
            tasks.append((key, f.read(), refs[pid]))
    return tasks[:limit] if limit else tasks


def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["key"]] = rec
    return out


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
    ap.add_argument("--timing-from", default=None,
                    help="JSONL from a prior correctness pass. Only time the "
                         "passed keys, in this process. Timing must be a fresh "
                         "container: CUDA in the screening container can die "
                         "after the first worker pool exits.")
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

    prior = load_jsonl(args.timing_from) if args.timing_from else None
    only_keys = set(k for k, r in prior.items() if r.get("passed")) if prior else None
    tasks = load_candidates(args.kernel_dir, args.level, refs, args.limit,
                            only_keys=only_keys)
    print("verifying %d candidates with %d workers over %d GPUs"
          % (len(tasks), args.workers, args.gpus))

    def write_out(results):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            for key, rec in results.items():
                f.write(json.dumps(rec) + "\n")
        os.replace(tmp, args.out)

    start = time.time()
    if args.timing_from:
        results = prior
        passed = tasks
        print("timing %d survivors from %s on %d exclusive GPUs"
              % (len(passed), args.timing_from, args.gpus))
        if not passed:
            print("no passed candidates to time")
        else:
            with VerifierPool(workers=args.gpus, gpus=args.gpus,
                              num_correct_trials=args.num_correct_trials,
                              timeout_s=args.timeout, measure_time=True,
                              num_perf_trials=args.num_perf_trials,
                              ref_mode=args.ref_mode) as pool:
                timed = pool.verify_batch(passed)
            for key, rec in timed.items():
                if rec.get("passed"):
                    results[key] = rec
            write_out(results)
    elif args.measure_time:
        results = verify_and_time(
            tasks, workers=args.workers, gpus=args.gpus,
            num_correct_trials=args.num_correct_trials, timeout_s=args.timeout,
            num_perf_trials=args.num_perf_trials, progress=print,
            ref_mode=args.ref_mode, checkpoint=write_out)
    else:
        with VerifierPool(workers=args.workers, gpus=args.gpus,
                          num_correct_trials=args.num_correct_trials,
                          timeout_s=args.timeout) as pool:
            results = pool.verify_batch(tasks)

    n_pass = sum(r["passed"] for r in results.values())
    n_skip = sum(r["stage"] in INCONCLUSIVE_STAGES for r in results.values())

    write_out(results)

    elapsed = time.time() - start
    print("done: %d/%d passed (%.1f%%), %d inconclusive in %.1fs -> %.1f cand/s"
          % (n_pass, len(tasks), n_pass / max(len(tasks), 1) * 100,
             n_skip, elapsed, len(tasks) / max(elapsed, 1e-9)))
    if n_skip:
        print("  inconclusive (oom / cuda_poison / worker_crash) is retried "
              "inside the pool; leftovers are not exec failures")

    speedups = sorted(r["speedup"] for r in results.values()
                      if r.get("speedup"))
    if speedups:
        faster = sum(1 for s in speedups if s > 1.0)
        print("  speed: median %.3fx, %d/%d beat the reference"
              % (speedups[len(speedups) // 2], faster, len(speedups)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
