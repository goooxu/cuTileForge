"""Fast correctness-only verifier for generated cuTile kernels.

The benchmark harness (scripts/eval_from_generations.py) runs at roughly 6
samples/min on 4 GPUs, which is far too slow for the tens of thousands of
candidates rejection sampling needs. Measurement showed cuTile's JIT compile is
only ~65 ms, so that cost is dominated by per-sample process startup and the
100 performance trials -- neither of which rejection sampling needs.

This keeps the benchmark's pass criterion exactly (numerically correct AND
implemented entirely in cuTile, evaluated in true fp32) but drops the timing and
amortises interpreter startup across a persistent worker pool.

Usage:
    python3 verify/fast_verify.py --kernel-dir runs/synth/kernels \\
        --level 90 --out runs/synth/verified.jsonl --workers 16
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import signal
import time

# Set before torch is imported in any worker.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")


def _worker(task_q: "mp.Queue", result_q: "mp.Queue", device_id: int,
            num_correct_trials: int, timeout_s: float) -> None:
    """Persistent worker: import torch once, then verify candidates forever."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    import torch
    from kernelbench.eval import load_original_model_and_inputs, set_seed
    from kernelbench.utils import enforce_reference_precision
    from kernelbench.kernel_static_checker import (
        check_cutile_impl, check_pytorch_wrap, check_torch_computation_ops,
    )

    enforce_reference_precision()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    import importlib.util
    import tempfile

    class Timeout(Exception):
        pass

    def _on_alarm(signum, frame):
        raise Timeout("exceeded %.0fs" % timeout_s)

    # A generated kernel can loop forever, which would otherwise wedge this
    # worker for the rest of the run.
    signal.signal(signal.SIGALRM, _on_alarm)

    def purity(code: str):
        """Same gate as the benchmark: real cuTile, and nothing left in torch."""
        bad, msg = check_cutile_impl(code)
        if bad:
            return False, msg
        bad, msg = check_torch_computation_ops(code)
        if bad:
            return False, msg
        bad, msg = check_pytorch_wrap(code)
        if bad:
            return False, msg
        return True, ""

    def load_model_new(code: str):
        """cuTile's @ct.kernel does not survive exec(), so go via a real module."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            spec = importlib.util.spec_from_file_location("cand_%d" % os.getpid(), path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.ModelNew, path
        except Exception:
            os.unlink(path)
            raise

    while True:
        item = task_q.get()
        if item is None:
            break

        key, code, ref_src = item
        rec = {"key": key, "passed": False, "stage": "", "error": ""}
        tmp_path = None
        t0 = time.perf_counter()
        signal.alarm(int(timeout_s))
        try:
            ok, msg = purity(code)
            if not ok:
                rec["stage"] = "purity"
                rec["error"] = msg
                result_q.put(rec)
                continue

            ctx = {}
            Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_src, ctx)

            ModelNew, tmp_path = load_model_new(code)

            with torch.no_grad():
                set_seed(0)
                init_inputs = [x.cuda() if hasattr(x, "cuda") else x
                               for x in get_init_inputs()]
                ref_model = Model(*init_inputs).cuda()
                new_model = ModelNew(*init_inputs).cuda()

                for trial in range(num_correct_trials):
                    # Randomised inputs per trial: a kernel that hardcodes an
                    # answer for one draw should not survive.
                    set_seed(trial + 1)
                    inputs = [x.cuda() if hasattr(x, "cuda") else x
                              for x in get_inputs()]
                    expected = ref_model(*inputs)
                    got = new_model(*inputs)
                    torch.cuda.synchronize()

                    if got.shape != expected.shape:
                        raise AssertionError(
                            "shape %s != expected %s" % (tuple(got.shape),
                                                         tuple(expected.shape)))
                    if not torch.isfinite(got).all():
                        raise AssertionError("output contains non-finite values")
                    if not torch.allclose(got, expected, atol=1e-4, rtol=1e-4):
                        raise AssertionError(
                            "output mismatch, max diff %.4g"
                            % (got - expected).abs().max().item())

            rec["passed"] = True
            rec["stage"] = "pass"
        except Timeout as e:
            rec["stage"] = "timeout"
            rec["error"] = str(e)
        except torch.cuda.OutOfMemoryError as e:
            # Several workers share each GPU, so OOM says nothing about the
            # candidate. Flag it separately instead of counting it as a failure.
            rec["stage"] = "oom"
            rec["error"] = str(e)[:200]
        except Exception as e:
            rec["stage"] = "exec"
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        finally:
            signal.alarm(0)
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            # Keep peak memory from creeping up across candidates.
            torch.cuda.empty_cache()
            rec["seconds"] = round(time.perf_counter() - t0, 3)
            result_q.put(rec)


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
    args = ap.parse_args()

    from kernelbench.dataset import construct_kernelbench_dataset
    dataset = construct_kernelbench_dataset(args.level)
    refs = {p_id: dataset.get_problem_by_id(p_id).code
            for p_id in dataset.get_problem_ids()}

    pattern = re.compile(r"level_%d_problem_(\d+)_sample_(\d+)_kernel\.py" % args.level)
    tasks = []
    for fname in sorted(os.listdir(args.kernel_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        pid, sid = int(m.group(1)), int(m.group(2))
        if pid not in refs:
            continue
        with open(os.path.join(args.kernel_dir, fname)) as f:
            code = f.read()
        tasks.append(("%d:%d" % (pid, sid), code, refs[pid]))
    if args.limit:
        tasks = tasks[:args.limit]

    print("verifying %d candidates with %d workers over %d GPUs"
          % (len(tasks), args.workers, args.gpus))

    ctx = mp.get_context("spawn")
    task_q, result_q = ctx.Queue(), ctx.Queue()
    procs = []
    for i in range(args.workers):
        p = ctx.Process(target=_worker,
                        args=(task_q, result_q, i % args.gpus,
                              args.num_correct_trials, args.timeout),
                        daemon=True)
        p.start()
        procs.append(p)

    for t in tasks:
        task_q.put(t)
    for _ in procs:
        task_q.put(None)

    n_pass = n_oom = 0
    start = time.time()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(len(tasks)):
            rec = result_q.get()
            n_pass += rec["passed"]
            n_oom += rec["stage"] == "oom"
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - start)
                print("  %d/%d  passed %d  oom %d  %.1f cand/s"
                      % (i + 1, len(tasks), n_pass, n_oom, rate))

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    elapsed = time.time() - start
    print("done: %d/%d passed (%.1f%%), %d OOM (inconclusive) in %.1fs -> %.1f cand/s"
          % (n_pass, len(tasks), n_pass / max(len(tasks), 1) * 100,
             n_oom, elapsed, len(tasks) / max(elapsed, 1e-9)))
    if n_oom:
        print("  rerun OOM candidates with fewer --workers to resolve them")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
