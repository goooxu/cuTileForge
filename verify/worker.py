"""Reusable persistent verification pool for generated cuTile kernels.

Extracted from fast_verify.py so the repair loop can verify a candidate and get
an answer back immediately, instead of only running as a batch over a directory.

The pass criterion is deliberately identical to the benchmark's: numerically
correct AND implemented entirely in cuTile, evaluated in true fp32. Changing it
here would silently break comparability with every number measured so far.

Typical use:

    with VerifierPool(workers=16, gpus=4) as pool:
        results = pool.verify_batch([(key, code, ref_src), ...])
"""

import multiprocessing as mp
import os
import signal
import time

# Must be set before any worker imports torch.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

STOP = None


def _median(xs):
    """Median rather than mean: one descheduled trial should not set the number."""
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def _worker(task_q, result_q, device_id: int, num_correct_trials: int,
            timeout_s: float, measure_time: bool = False,
            num_perf_trials: int = 20) -> None:
    """Import torch once, then verify candidates until told to stop.

    With measure_time, a candidate that passes is also timed against the
    reference. That must only be used by a pool with one worker per GPU: several
    workers sharing a device contaminate each other's measurements, and a
    speedup measured under contention is not a speedup.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    import importlib.util
    import tempfile

    import torch
    from kernelbench.eval import load_original_model_and_inputs, set_seed
    from kernelbench.utils import enforce_reference_precision
    from kernelbench.kernel_static_checker import (
        check_cutile_impl, check_pytorch_wrap, check_torch_computation_ops,
    )
    # The benchmark's own timing function, deliberately: warmup, L2 flush and
    # CUDA-event bracketing all have to match or the training signal and the
    # reported number stop meaning the same thing.
    from kernelbench.timing import time_execution_with_cuda_event

    enforce_reference_precision()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    class Timeout(Exception):
        pass

    def _on_alarm(signum, frame):
        raise Timeout("exceeded %.0fs" % timeout_s)

    # A generated kernel can loop forever, which would otherwise wedge this
    # worker for the rest of the run.
    signal.signal(signal.SIGALRM, _on_alarm)

    def purity(code: str):
        """Same gate as the benchmark: real cuTile, and nothing left in torch."""
        for check in (check_cutile_impl, check_torch_computation_ops,
                      check_pytorch_wrap):
            bad, msg = check(code)
            if bad:
                return False, msg
        return True, ""

    def load_model_new(code: str):
        """cuTile's @ct.kernel does not survive exec(), so go via a real module."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            spec = importlib.util.spec_from_file_location(
                "cand_%d_%d" % (os.getpid(), int(time.time() * 1e6) % 10 ** 9), path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.ModelNew, path
        except Exception:
            os.unlink(path)
            raise

    while True:
        item = task_q.get()
        if item is STOP:
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
                continue

            ctx = {}
            Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                ref_src, ctx)
            ModelNew, tmp_path = load_model_new(code)

            with torch.no_grad():
                set_seed(0)
                init_inputs = [x.cuda() if hasattr(x, "cuda") else x
                               for x in get_init_inputs()]
                # The seed must be reset before *each* construction, matching
                # KernelBench's protocol. Seeding once lets the reference consume
                # RNG while building its parameters, so ModelNew would draw
                # different weights and could never match -- which silently makes
                # every task that owns learnable parameters unpassable.
                set_seed(0)
                ref_model = Model(*init_inputs).cuda()
                set_seed(0)
                new_model = ModelNew(*init_inputs).cuda()

                for trial in range(num_correct_trials):
                    # Fresh inputs per trial: a kernel that hardcodes an answer
                    # for one draw should not survive.
                    set_seed(trial + 1)
                    inputs = [x.cuda() if hasattr(x, "cuda") else x
                              for x in get_inputs()]
                    expected = ref_model(*inputs)
                    got = new_model(*inputs)
                    torch.cuda.synchronize()

                    if got.shape != expected.shape:
                        raise AssertionError(
                            "shape %s != expected %s"
                            % (tuple(got.shape), tuple(expected.shape)))
                    if not torch.isfinite(got).all():
                        raise AssertionError("output contains non-finite values")
                    if not torch.allclose(got, expected, atol=1e-4, rtol=1e-4):
                        # Record how wrong, not just that it is wrong. 92% of the
                        # benchmark problems the best model cannot solve fail
                        # here -- pure cuTile that compiles, launches and returns
                        # the right shape with the wrong values -- and a flat
                        # verdict gives a reward function nothing to grade them
                        # by. Scaled by the reference's own magnitude so the
                        # number means the same thing across operators.
                        diff = (got - expected).abs().max().item()
                        scale = expected.abs().max().item()
                        rec["max_diff"] = diff
                        rec["rel_diff"] = diff / scale if scale > 0 else diff
                        raise AssertionError(
                            "output mismatch, max diff %.4g" % diff)

            rec["passed"] = True
            rec["stage"] = "pass"

            if measure_time:
                # Time the reference here rather than reading a cached baseline
                # file: it costs one extra measurement but keeps both sides on
                # the same device in the same thermal and clock state, and means
                # synthetic task sets need no baseline artefact of their own.
                with torch.no_grad():
                    set_seed(1)
                    inputs = [x.cuda() if hasattr(x, "cuda") else x
                              for x in get_inputs()]
                    ref_ms = _median(time_execution_with_cuda_event(
                        ref_model, inputs, num_warmup=5, num_trials=num_perf_trials,
                        verbose=False, device=device))
                    new_ms = _median(time_execution_with_cuda_event(
                        new_model, inputs, num_warmup=5, num_trials=num_perf_trials,
                        verbose=False, device=device))
                rec["ref_ms"] = round(ref_ms, 5)
                rec["kernel_ms"] = round(new_ms, 5)
                rec["speedup"] = round(ref_ms / new_ms, 4) if new_ms > 0 else None
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
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        finally:
            signal.alarm(0)
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            torch.cuda.empty_cache()
            rec["seconds"] = round(time.perf_counter() - t0, 3)
            result_q.put(rec)


class VerifierPool:
    """Persistent pool of GPU verification workers.

    Interpreter and CUDA-context startup dominate per-candidate cost, so the
    workers are kept alive across batches rather than respawned.

    Workers are treated as expendable. A generated kernel can corrupt the CUDA
    context badly enough to kill its worker outright -- an illegal instruction
    inside even torch.cuda.empty_cache() -- and the task it was holding then
    never produces a result. Waiting on the queue for it deadlocks the run, so
    the pool watches for stalls, respawns dead workers, and retries whatever
    went missing.
    """

    def __init__(self, workers: int = 16, gpus: int = 4,
                 num_correct_trials: int = 2, timeout_s: float = 120.0,
                 measure_time: bool = False, num_perf_trials: int = 20):
        if measure_time and workers > gpus:
            raise ValueError(
                "timing needs exclusive GPUs: %d workers over %d gpus would have "
                "them contend and the speedups would be meaningless"
                % (workers, gpus))
        self.workers = workers
        self.gpus = gpus
        self.num_correct_trials = num_correct_trials
        self.timeout_s = timeout_s
        self.measure_time = measure_time
        self.num_perf_trials = num_perf_trials
        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.procs = [None] * workers
        for i in range(workers):
            self._spawn(i)

    def _spawn(self, i: int) -> None:
        p = self.ctx.Process(target=_worker,
                             args=(self.task_q, self.result_q, i % self.gpus,
                                   self.num_correct_trials, self.timeout_s,
                                   self.measure_time, self.num_perf_trials),
                             daemon=True)
        p.start()
        self.procs[i] = p

    def _respawn_dead(self) -> int:
        n = 0
        for i, p in enumerate(self.procs):
            if p is not None and not p.is_alive():
                self._spawn(i)
                n += 1
        return n

    def _run_once(self, pending: dict) -> dict:
        """Dispatch pending {key: item} and collect what comes back."""
        import queue as _queue

        for it in pending.values():
            self.task_q.put(it)

        out = {}
        # A worker killed mid-task is only detectable as silence, so allow one
        # full candidate timeout plus slack before concluding the batch stalled.
        stall_s = self.timeout_s + 60
        last = time.time()
        while len(out) < len(pending):
            try:
                rec = self.result_q.get(timeout=5)
            except _queue.Empty:
                if time.time() - last > stall_s:
                    break
                continue
            out[rec["key"]] = rec
            last = time.time()
        return out

    def verify_batch(self, items, max_retries: int = 2) -> dict:
        """Verify (key, code, ref_src) triples; return {key: result}.

        Results come back out of order, so they are keyed rather than listed.
        OOM is retried rather than reported: workers share each GPU, so it says
        nothing about the candidate and must not be mistaken for a verdict.
        """
        pending = {key: (key, code, ref) for key, code, ref in items}
        out = {}

        for attempt in range(max_retries + 1):
            if not pending:
                break
            got = self._run_once(pending)
            out.update(got)

            crashed = {k: v for k, v in pending.items() if k not in got}
            oom = {k: pending[k] for k, r in got.items() if r["stage"] == "oom"}
            self._respawn_dead()

            pending = {}
            pending.update(crashed)
            pending.update(oom)
            if pending and attempt < max_retries:
                # Let transient pressure clear before trying these again.
                time.sleep(5)

        for key, item in pending.items():
            out.setdefault(key, {"key": key, "passed": False,
                                 "stage": "worker_crash",
                                 "error": "verifier worker died on this candidate",
                                 "seconds": 0.0})
            if out[key]["stage"] == "oom":
                out[key] = {"key": key, "passed": False, "stage": "oom",
                            "error": "out of memory after retries", "seconds": 0.0}
        return out

    def close(self) -> None:
        for _ in self.procs:
            self.task_q.put(STOP)
        for p in self.procs:
            if p is None:
                continue
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def verify_and_time(items, workers: int = 16, gpus: int = 4,
                    num_correct_trials: int = 2, timeout_s: float = 120.0,
                    num_perf_trials: int = 20, progress=None) -> dict:
    """Screen for correctness in parallel, then time the survivors exclusively.

    Timing and throughput pull in opposite directions. Correctness screening
    wants every GPU oversubscribed; timing wants each GPU to itself. Since only
    a minority of candidates are correct, running them as two phases costs
    little: the expensive phase only sees what survived the cheap one.

    Returns {key: record}, where records that reached timing also carry
    speedup, ref_ms and kernel_ms.
    """
    items = list(items)
    by_key = {k: (k, c, r) for k, c, r in items}

    with VerifierPool(workers=workers, gpus=gpus,
                      num_correct_trials=num_correct_trials,
                      timeout_s=timeout_s) as pool:
        results = pool.verify_batch(items)

    survivors = [by_key[k] for k, r in results.items() if r["passed"]]
    if progress:
        progress("correct: %d/%d, timing survivors on %d exclusive GPUs"
                 % (len(survivors), len(items), gpus))
    if not survivors:
        return results

    with VerifierPool(workers=gpus, gpus=gpus,
                      num_correct_trials=num_correct_trials,
                      timeout_s=timeout_s, measure_time=True,
                      num_perf_trials=num_perf_trials) as pool:
        timed = pool.verify_batch(survivors)

    for key, rec in timed.items():
        # A candidate that passed the screen but failed here is a flake, not a
        # verdict; keep the screening result and leave it without a speedup.
        if rec["passed"]:
            results[key] = rec
    return results
