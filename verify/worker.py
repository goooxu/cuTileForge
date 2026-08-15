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
import queue
import re
import signal
import time

# Must be set before any worker imports torch.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

STOP = None

# A generated kernel can leave the GPU in a sticky error state. Later
# candidates on the same device then fail in a millisecond without running.
# empty_cache does not clear that; the worker has to reset the device and die
# so the pool can retry on a fresh context. OOM is already inconclusive; these
# are the same class of harness failure.
_CUDA_CONTEXT_MARKERS = (
    "illegal memory access",
    "illegal instruction",
    "unspecified launch failure",
    "cudaerrorillegaladdress",
    "cudaerrorillegalinstruction",
)
INCONCLUSIVE_STAGES = ("oom", "cuda_poison", "worker_crash")
# Retry these mid-batch. worker_crash is only assigned after retries are spent.
_RETRY_STAGES = ("oom", "cuda_poison")
# Instant failures are contagion. A real launch+fault is a few hundred ms.
_POISON_RETRY_SECONDS = 0.05


def is_cuda_context_error(exc_or_msg):
    """True when the GPU context is no longer usable for the next candidate."""
    text = str(exc_or_msg).lower()
    if "out of memory" in text:
        return False
    return any(m in text for m in _CUDA_CONTEXT_MARKERS)


def reset_cuda_device(torch_mod):
    """Destroy the current CUDA context. Sibling processes on this GPU must die."""
    try:
        torch_mod.cuda.synchronize()
    except Exception:
        pass
    try:
        torch_mod.cuda.cudart().cudaDeviceReset()
    except Exception:
        pass


def _reset_gpu_proc(device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch
    reset_cuda_device(torch)


def _drain(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _stop_proc(p, timeout=5):
    if p is None:
        return
    if p.is_alive():
        p.terminate()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()


def _median(xs):
    """Median rather than mean: one descheduled trial should not set the number."""
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def _worker(task_q, result_q, device_id: int, num_correct_trials: int,
            timeout_s: float, measure_time: bool = False,
            num_perf_trials: int = 20, ref_mode: str = "compile") -> None:
    """Import torch once, then verify candidates until told to stop.

    With measure_time, a candidate that passes is also timed against the
    reference. That must only be used by a pool with one worker per GPU: several
    workers sharing a device contaminate each other's measurements, and a
    speedup measured under contention is not a speedup.

    ref_mode decides what the reference is. "compile" runs it under
    torch.compile, which is the honest comparison and the default: half the
    benchmark is fusion chains, and beating eager there mostly means beating
    intermediate materialisation, which inductor already removes. Measured
    against eager the best model looked like it matched PyTorch at the median
    (1.00x); against compile it is 0.92x. "eager" is kept for reproducing the
    older numbers.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    import importlib.util
    import tempfile

    import torch
    from kernelbench.eval import load_original_model_and_inputs, set_seed
    from kernelbench.utils import enforce_reference_precision
    from kernelbench.kernel_static_checker import (
        TORCH_COMPUTATION_OPS, TORCH_FUNCTIONAL_PATTERNS, _strip_comments,
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

    # Compiling the reference costs a couple of seconds, and every candidate for
    # a given task shares the same reference -- so compile once per task and
    # reuse it for the rest of this worker's life. Keyed on the reference source,
    # since that is what defines the model.
    _ref_cache = {}

    def _compiled_ref(ref_src: str, eager_model, make_inputs):
        key = hash(ref_src)
        if key not in _ref_cache:
            try:
                m = torch.compile(eager_model)
                # Force the compile now, inside this task's alarm, so a model
                # inductor cannot handle fails here rather than polluting the
                # timing loop.
                with torch.no_grad():
                    set_seed(1)
                    warm = [x.cuda() if hasattr(x, "cuda") else x
                            for x in make_inputs()]
                    m(*warm)
                torch.cuda.synchronize()
                _ref_cache[key] = (m, "compile")
            except Exception:
                # Some references do not compile. Falling back to eager is right,
                # but the record has to say so: a speedup against eager and one
                # against inductor are different numbers.
                _ref_cache[key] = (eager_model, "eager_compile_failed")
        return _ref_cache[key]

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

    def purity_detail(code: str):
        """Why the gate failed, in enough detail to grade the failure.

        The largest group of benchmark problems the best model cannot solve is
        this one: a numerically correct kernel rejected because one pointwise
        activation was left in torch. Of 73 such samples the leftovers were
        torch.relu 17 times, torch.softmax 12, torch.sigmoid 11, F.gelu 8 -- the
        model ports the convolution and the norm and then finishes the chain with
        torch.relu(...). A single flat verdict cannot tell that from a wholesale
        passthrough, so the reward cannot either.

        Returns (has_real_kernel, delegates_to_nn, torch_ops_left). The op list
        is imported from the checker rather than restated, so the gate and this
        stay in step; the checker itself is untouched, since it defines the
        benchmark's verdict and must not drift.
        """
        stripped = _strip_comments(code)
        has_kernel = not check_cutile_impl(code)[0]
        delegates = bool(check_pytorch_wrap(code)[0])

        n = len(re.findall(
            r"\b(" + "|".join(re.escape(f) for f in TORCH_COMPUTATION_OPS)
            + r")(?=\s*\(|\s|$)", stripped))
        for pattern in TORCH_FUNCTIONAL_PATTERNS:
            n += len(re.findall(pattern, stripped))
        return has_kernel, delegates, n

    def impure_numerics(code: str, ref_src: str):
        """Does a kernel that failed the purity gate at least compute the answer?

        Deliberately separate from the main verification path rather than
        threading a flag through it: everything downstream depends on that path,
        and this is a side question. It costs one extra run on about 4% of
        candidates -- the ones that failed the gate while still containing a real
        launched kernel.

        Worth the duplication because the distinction it draws is the largest
        remaining one. A chain whose heavy part is correctly in cuTile and whose
        tail stayed as torch.relu is one edit from passing; a chain that is impure
        *and* wrong is not, and the reward should not pay them the same.
        """
        path = None
        try:
            ctx = {}
            Model, get_init, get_in = load_original_model_and_inputs(ref_src, ctx)
            ModelNew, path = load_model_new(code)
            with torch.no_grad():
                set_seed(0)
                init_inputs = [x.cuda() if hasattr(x, "cuda") else x
                               for x in get_init()]
                set_seed(0)
                ref_m = Model(*init_inputs).cuda()
                set_seed(0)
                new_m = ModelNew(*init_inputs).cuda()
                set_seed(1)
                inputs = [x.cuda() if hasattr(x, "cuda") else x for x in get_in()]
                expected, got = ref_m(*inputs), new_m(*inputs)
                torch.cuda.synchronize()
                return (got.shape == expected.shape
                        and bool(torch.isfinite(got).all())
                        and bool(torch.allclose(got, expected,
                                                atol=1e-4, rtol=1e-4)))
        except Exception as e:
            if is_cuda_context_error(e):
                raise
            return False
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

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
        poison = False
        t0 = time.perf_counter()
        signal.alarm(int(timeout_s))
        try:
            ok, msg = purity(code)
            if not ok:
                rec["stage"] = "purity"
                rec["error"] = msg
                has_kernel, delegates, n_torch = purity_detail(code)
                rec["has_real_kernel"] = has_kernel
                rec["delegates_to_nn"] = delegates
                rec["torch_ops_left"] = n_torch
                # Only for plausibly-unfinished ports. Wholesale delegation is
                # already scored zero, so its numerics are not worth a GPU run.
                if has_kernel and not delegates:
                    rec["impure_correct"] = impure_numerics(code, ref_src)
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
                timed_ref, mode = ref_model, "eager"
                if ref_mode == "compile":
                    timed_ref, mode = _compiled_ref(ref_src, ref_model, get_inputs)
                rec["ref_mode"] = mode
                with torch.no_grad():
                    set_seed(1)
                    inputs = [x.cuda() if hasattr(x, "cuda") else x
                              for x in get_inputs()]
                    ref_ms = _median(time_execution_with_cuda_event(
                        timed_ref, inputs, num_warmup=5, num_trials=num_perf_trials,
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
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
            if is_cuda_context_error(e):
                # Sticky on the device. Put the result first: empty_cache after
                # an illegal instruction can kill the process and lose the key.
                rec["stage"] = "cuda_poison"
                poison = True
            else:
                rec["stage"] = "exec"
        finally:
            signal.alarm(0)
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            rec["seconds"] = round(time.perf_counter() - t0, 3)
            result_q.put(rec)
            if poison:
                # Do not reset here: sibling workers on this GPU are mid-call
                # and would hang. The pool kills them, resets each device, then
                # respawns.
                os._exit(1)
            try:
                torch.cuda.empty_cache()
            except Exception:
                os._exit(1)


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

    A sticky illegal-memory-access is worse: the worker survives, but every
    later candidate on that GPU fails in a millisecond. Those are marked
    cuda_poison, the device is reset, every worker is recycled, and the
    candidates are retried. A kernel that still faults after a fresh context
    is an exec failure.
    """

    def __init__(self, workers: int = 16, gpus: int = 4,
                 num_correct_trials: int = 2, timeout_s: float = 120.0,
                 measure_time: bool = False, num_perf_trials: int = 20,
                 ref_mode: str = "compile"):
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
        self.ref_mode = ref_mode
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
                                   self.measure_time, self.num_perf_trials,
                                   self.ref_mode),
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

    def _recycle_all(self) -> None:
        """Kill every worker, reset each GPU, then spawn a clean pool.

        Resetting from inside a live worker hangs siblings that are mid-call.
        The device error is sticky, so new workers must start after a reset.
        """
        for p in self.procs:
            _stop_proc(p)
        _drain(self.task_q)
        _drain(self.result_q)
        resets = []
        for gpu in range(self.gpus):
            p = self.ctx.Process(target=_reset_gpu_proc, args=(gpu,), daemon=True)
            p.start()
            resets.append(p)
        for p in resets:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        for i in range(self.workers):
            self._spawn(i)

    def _run_once(self, pending: dict) -> dict:
        """Dispatch pending {key: item} and collect what comes back."""
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
            except queue.Empty:
                if time.time() - last > stall_s:
                    break
                continue
            if rec.get("key") not in pending:
                continue
            out[rec["key"]] = rec
            last = time.time()
        return out

    def verify_batch(self, items, max_retries: int = 2) -> dict:
        """Verify (key, code, ref_src) triples; return {key: result}.

        Results come back out of order, so they are keyed rather than listed.
        OOM and a sticky CUDA context are retried rather than reported: both
        say nothing about the candidate. A kernel that still illegal-accesses
        after a fresh worker is an exec failure, not inconclusive.
        """
        pending = {key: (key, code, ref) for key, code, ref in items}
        out = {}

        for attempt in range(max_retries + 1):
            if not pending:
                break
            got = self._run_once(pending)
            out.update(got)

            crashed = {k: v for k, v in pending.items() if k not in got}
            retry = {k: pending[k] for k, r in got.items()
                     if r["stage"] in _RETRY_STAGES}
            if any(r["stage"] == "cuda_poison" for r in got.values()):
                self._recycle_all()
            else:
                self._respawn_dead()

            pending = {}
            pending.update(crashed)
            pending.update(retry)
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
        for rec in out.values():
            if rec.get("stage") != "cuda_poison":
                continue
            # Survived retries and still poisoned: a real IMA takes hundreds of
            # ms, contagion takes 1 ms. Only the former is the kernel's fault.
            if rec.get("seconds", 0) >= _POISON_RETRY_SECONDS:
                rec["stage"] = "exec"
            else:
                rec["error"] = (rec.get("error") or "") + " (after retries)"
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
                    num_perf_trials: int = 20, progress=None,
                    ref_mode: str = "compile", checkpoint=None) -> dict:
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
    if checkpoint:
        checkpoint(results)
        if progress:
            progress("checkpointed correctness before timing")
    if not survivors:
        return results

    with VerifierPool(workers=gpus, gpus=gpus,
                      num_correct_trials=num_correct_trials,
                      timeout_s=timeout_s, measure_time=True,
                      num_perf_trials=num_perf_trials,
                      ref_mode=ref_mode) as pool:
        timed = pool.verify_batch(survivors)

    for key, rec in timed.items():
        # A candidate that passed the screen but failed here is a flake, not a
        # verdict; keep the screening result and leave it without a speedup.
        if rec["passed"]:
            results[key] = rec
    if checkpoint:
        checkpoint(results)
    return results
