"""Score a cuTile KernelBench run and classify what went wrong.

A sample counts as a pass only if it is numerically correct *and* implemented
entirely in cuTile. KernelBench itself permits leaving operators in PyTorch, so
by its own rules a ModelNew that just calls torch.matmul scores as correct at
~1x speedup while containing no cuTile at all, and a partial port scores as a
full pass. Neither answers "can this model write cuTile", so both are counted as
failures here.

"Entirely in cuTile" means all three of:
  * a cuTile kernel is defined and actually dispatched (check_cutile_impl)
  * no torch compute ops remain (check_torch_computation_ops)
  * no torch.nn compute layers remain (check_pytorch_wrap)

The latter two are KernelBench's own definitions, which already permit the host
scaffolding a cuTile launcher needs: nn.Module/Parameter, torch.empty_like,
.contiguous(), and so on.

Usage:
    python3 scripts/analyze_cutile_run.py --run-name cutile_l1 --level 1 --num-samples 8
"""

import argparse
import collections
import json
import os
import re

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.kernel_static_checker import (
    check_cutile_impl,
    check_pytorch_wrap,
    check_torch_computation_ops,
)

# Every public cuda.tile symbol; anything else referenced as ct.<name> is invented.
try:
    import cuda.tile as _ct
    CUTILE_SYMBOLS = {n for n in dir(_ct) if not n.startswith("_")}
except Exception:                                              # analysis host may lack the GPU stack
    CUTILE_SYMBOLS = set()


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Codex (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


# Writing another DSL's API is a property of the source, so these are matched
# against the generated code. Ordered; first match wins.
CODE_RULES = [
    ("triton_leakage",   r"@triton\.|\btl\.(load|store|program_id|arange|constexpr)\b|triton\.language"),
    ("cuda_cpp_leakage", r"__global__|load_inline|cpp_extension|threadIdx|blockIdx"),
    ("cute_leakage",     r"cute::|cutlass::|from cutlass|import cutlass"),
]

# Diagnosed from the compiler/runtime message. Matching these against the source
# would misfire constantly: a kernel that merely mentions "dtype" is not thereby
# a dtype error. Ordered; first match wins.
MSG_RULES = [
    # cuTile grids are at most 3D. The model reaches for one block per output
    # element on 4D/5D NCHW/NCDHW tensors, which cuTile simply cannot express.
    ("grid_rank_exceeded",   r"Grid dimensions must be at most 3"),
    # Treating a cuTile Array as if it were a torch tensor.
    ("array_used_as_tensor", r"not directly subscriptable|No such attribute '(view|reshape|size|shape|numel|permute|transpose|contiguous|t)'"),
    ("rank_mismatch",        r"Expected shape length to be|Index size \d+ does not match|rank and tile rank to match|does not match the array rank|Axis must be 0, 1, or 2|Axis \d+ is out of range"),
    ("matmul_shape",         r"Incompatible shapes for matrix mul|Inner dimensions must match"),
    ("kernel_call_convention", r"Tile kernels cannot be called directly|Tile functions can only be called from tile code"),
    # ct.launch cannot take None. KernelBench models carry optional parameters
    # (bias=None, stride=None) that the model forwards straight into the kernel.
    ("none_kernel_argument", r"Unsupported argument type NoneType|'NoneType' object"),
    ("tile_indexing",        r"Directly indexing a tile|Tiles are immutable|np\.newaxis"),
    ("loop_break",           r"Break in a for loop is not supported"),
    ("host_python_in_kernel", r"Python value .* is not supported|unsupported operand type"),
    ("type_depends_on_path", r"depends on path taken"),
    ("acc_shape_mismatch",   r"Expect acc shape to be"),
    ("undefined_name",       r"Undefined variable|is not defined|NameError"),
    ("python_syntax",        r"SyntaxError|IndentationError|invalid syntax|unexpected indent|prior to global declaration"),
    ("tile_shape_invalid",   r"power[- ]of[- ]two|must be a power|not a power of 2|shape is too big"),
    ("shape_not_constant",   r"compile[- ]time constant|not a constant|must be constant|non-constant|expects a constant argument"),
    ("host_shape_error",     r"invalid for input of size|Kernel expects \d+ arguments"),
    ("wrong_arg_type",       r"Invalid argument|Expected an array|Expected a tile"),
    ("dtype_mismatch",       r"dtype|type mismatch|incompatible type"),
    ("bad_launch",           r"positional argument|takes \d+ .*argument|missing \d+ required|unexpected keyword"),
    ("unsupported_feature",  r"not supported|unsupported"),
    ("attribute_error",      r"has no attribute|AttributeError"),
    ("oom",                  r"out of memory"),
    ("timeout",              r"timeout|timed out"),
]


def error_text(meta: dict) -> str:
    """Error message regardless of which stage recorded it.

    Import-stage failures land in `compilation_error`, launch/run failures in
    `runtime_error`, and harness-level problems in `error`.
    """
    for key in ("runtime_error", "compilation_error", "other_error", "error"):
        val = meta.get(key)
        if val:
            return str(val)
    return ""

# Which stage the sample died at. For cuTile, KernelBench's `compiled` flag only
# means the module imported: @ct.kernel does not compile eagerly, so tileiras
# actually runs at first launch, inside the correctness check.
def classify_stage(res: dict, evaluated: bool, generated: bool) -> str:
    if not generated:
        return "no_code_generated"
    if not evaluated:
        return "not_evaluated"
    meta = res.get("metadata") or {}
    if not res.get("compiled", False):
        return "import_error"
    exc = str(meta.get("runtime_error_name", ""))
    if exc:
        if "TileCompilerExecutionError" in exc or "TileCompilerTimeoutError" in exc:
            return "cutile_backend_compile_error"
        if "cuda.tile" in exc:
            return "cutile_frontend_error"
        return "other_runtime_error"
    if meta.get("correctness_issue"):
        return "wrong_numerics"
    return "other"


def classify_error(res: dict, code: str, evaluated: bool, generated: bool) -> str:
    if not generated:
        return "no_code_generated"
    if not evaluated:
        return "not_evaluated"
    for name, pattern in CODE_RULES:
        if re.search(pattern, code):
            return name
    meta = res.get("metadata") or {}
    msg = error_text(meta)
    if not msg and meta.get("correctness_issue"):
        return "wrong_numerics"
    for name, pattern in MSG_RULES:
        if re.search(pattern, msg, re.IGNORECASE):
            return name
    return "other"


def hallucinated_apis(code: str) -> set[str]:
    """cuda.tile attributes referenced by the code that do not exist."""
    if not CUTILE_SYMBOLS:
        return set()
    aliases = set(re.findall(r"import\s+cuda\.tile\s+as\s+(\w+)", code))
    aliases |= set(re.findall(r"from\s+cuda\s+import\s+tile\s+as\s+(\w+)", code))
    if not aliases:
        return set()
    used = set()
    for alias in aliases:
        used |= set(re.findall(rf"\b{re.escape(alias)}\.(\w+)", code))
    return used - CUTILE_SYMBOLS


def load_baselines(path: str, level: int) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get(f"level{level}", data.get(str(level), {}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/ws/runs")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--baseline", default=None,
                    help="Baseline timing JSON from generate_baseline_time.py.")
    ap.add_argument("--out", default=None, help="Write per-sample records as JSON.")
    args = ap.parse_args()

    run_dir = os.path.join(args.runs_dir, args.run_name)
    # Tolerate a missing or partial eval_results.json so the static findings
    # (cuTile usage, invented APIs, DSL leakage) can be read off while the GPU
    # evaluation is still running.
    eval_path = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
    else:
        print(f"[warn] no eval_results.json in {run_dir}; static analysis only")
        eval_results = {}

    dataset = construct_kernelbench_dataset(args.level)
    baselines = load_baselines(args.baseline, args.level)

    records = []
    for problem_id in dataset.get_problem_ids():
        problem = dataset.get_problem_by_id(problem_id)
        per_problem = {r["sample_id"]: r for r in eval_results.get(str(problem_id), [])}

        for sample_id in range(args.num_samples):
            kernel_path = os.path.join(
                run_dir,
                f"level_{args.level}_problem_{problem_id}_sample_{sample_id}_kernel.py",
            )
            code = ""
            if os.path.exists(kernel_path):
                with open(kernel_path) as f:
                    code = f.read()

            res = per_problem.get(sample_id, {})
            evaluated = sample_id in per_problem
            compiled = bool(res.get("compiled", False))
            correct = bool(res.get("correctness", False))
            meta = res.get("metadata", {}) or {}

            # Purity gate. Ordered so the reported reason is the most
            # fundamental one: no kernel at all beats "kernel plus leftovers".
            if not code:
                impure, gate_msg = True, "no code generated"
            else:
                impure, gate_msg = check_cutile_impl(code)
                if not impure:
                    impure, gate_msg = check_torch_computation_ops(code)
                if not impure:
                    impure, gate_msg = check_pytorch_wrap(code)
            fully_cutile = not impure

            passed = correct and fully_cutile

            speedup = None
            base = baselines.get(problem.name, {})
            base_ms = base.get("mean") if isinstance(base, dict) else None
            if passed and res.get("runtime", -1) > 0 and base_ms:
                speedup = base_ms / res["runtime"]

            records.append({
                "problem_id": problem_id,
                "problem": problem.name,
                "sample_id": sample_id,
                "generated": bool(code),
                "evaluated": evaluated,
                "compiled": compiled,
                "numerically_correct": correct,
                "fully_cutile": fully_cutile,
                "gate_reason": gate_msg,
                "passed": passed,
                "runtime_ms": res.get("runtime", -1),
                "baseline_ms": base_ms,
                "speedup": speedup,
                # Error classification describes the numerics/compile outcome, so
                # it is keyed on correctness rather than on the purity gate.
                "error_class": (None if correct
                                else classify_error(res, code, evaluated, bool(code))),
                "failure_stage": (None if correct
                                  else classify_stage(res, evaluated, bool(code))),
                "exception": (str(meta.get("runtime_error_name", ""))
                              or str(meta.get("compilation_error_name", "")) or None),
                "error_message": error_text(meta)[:300] or None,
                "hallucinated_apis": sorted(hallucinated_apis(code)),
            })

    report(records, args)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nwrote per-sample records to {args.out}")


def report(records: list[dict], args) -> None:
    n_samples = args.num_samples
    by_problem = collections.defaultdict(list)
    for r in records:
        by_problem[r["problem_id"]].append(r)

    total = len(records)
    n_problems = len(by_problem)

    def rate(key):
        return sum(r[key] for r in records) / total * 100 if total else 0.0

    print("=" * 78)
    print(f"cuTile KernelBench run: {args.run_name}  (level {args.level}, "
          f"{n_problems} problems x {n_samples} samples = {total})")
    print("=" * 78)

    # fast_p in KernelBench is the fraction of *problems* that are both correct
    # and faster than p, so take each problem's best passing sample.
    best_per_problem = {}
    for pid, rs in by_problem.items():
        cands = [r["speedup"] for r in rs if r["passed"] and r["speedup"]]
        if cands:
            best_per_problem[pid] = max(cands)

    sample_speedups = [r["speedup"] for r in records if r["passed"] and r["speedup"]]

    print("\n-- Headline --")
    print("  Correctness and speed are reported side by side. A kernel that is")
    print("  correct but slower than the library it replaces is not a kernel")
    print("  anyone would ship, and optimising correctness alone has been shown")
    print("  here to trade speed away: the fourth round's highest pass rate came")
    print("  with the lowest fast_1.0 of any run.")
    p1 = sum(pass_at_k(n_samples, sum(r["passed"] for r in rs), 1)
             for rs in by_problem.values()) / n_problems * 100
    print(f"  pass@1 (correct + entirely cuTile)   {p1:6.1f}%")
    if best_per_problem:
        f10 = sum(s > 1.0 for s in best_per_problem.values()) / n_problems * 100
        print(f"  fast_1.0 (also beats torch eager)    {f10:6.1f}%"
              f"   {sum(s > 1.0 for s in best_per_problem.values())}/{n_problems}"
              " problems")
    else:
        print("  fast_1.0                                n/a   (no --baseline given)")

    print("\n-- Per-sample rates --")
    print("  A sample passes only if it is numerically correct AND implemented")
    print("  entirely in cuTile; a partial port counts as a failure.")
    print(f"  generated                 {rate('generated'):6.1f}%")
    print(f"  entirely cuTile           {rate('fully_cutile'):6.1f}%")
    print(f"  module imported           {rate('compiled'):6.1f}%")
    print(f"  numerically correct       {rate('numerically_correct'):6.1f}%")
    print(f"  PASSED                    {rate('passed'):6.1f}%")

    print("\n-- pass@k --")
    for k in (1, 2, 4, 8):
        if k > n_samples:
            continue
        p = sum(pass_at_k(n_samples, sum(r["passed"] for r in rs), k)
                for rs in by_problem.values()) / n_problems * 100
        print(f"  pass@{k:<2} {p:6.1f}%")

    if sample_speedups:
        print("\n-- Speed vs torch eager (passing samples only) --")
        print("  fast_p over problems (best sample per problem):")
        for p in (0.5, 1.0, 2.0):
            frac = sum(s > p for s in best_per_problem.values()) / n_problems * 100
            print(f"    fast_{p:g}: {frac:5.1f}%")
        srt = sorted(sample_speedups)
        print(f"  per-sample speedup: median {srt[len(srt) // 2]:.3f}x  "
              f"max {max(srt):.2f}x  n={len(srt)}")
        faster = [s for s in sample_speedups if s > 1.0]
        print(f"  samples beating torch: {len(faster)}/{len(sample_speedups)}")
        print("  (per-category speed, and comparisons against other runs, come")
        print("   from train/compare_partial.py over the --out records)")
    else:
        print("\n-- Speed: no baseline supplied or no passing samples, skipping --")

    # Numerically-incorrect samples: what broke, independent of purity.
    fails = [r for r in records if not r["numerically_correct"]]

    print("\n-- Where samples died --")
    print("  (KernelBench's 'compiled' only means the module imported; @ct.kernel")
    print("   compiles lazily, so tileiras runs during the correctness check.)")
    for name, cnt in collections.Counter(r["failure_stage"] for r in fails).most_common():
        print(f"  {str(name):30s} {cnt:5d}  ({cnt / total * 100:5.1f}% of all samples)")

    print("\n-- Why samples failed --")
    for name, cnt in collections.Counter(r["error_class"] for r in fails).most_common():
        print(f"  {str(name):30s} {cnt:5d}  ({cnt / total * 100:5.1f}% of all samples)")

    excs = collections.Counter(r["exception"] for r in fails if r["exception"])
    if excs:
        print("\n-- Exception types raised --")
        for name, cnt in excs.most_common(12):
            print(f"  {cnt:5d}  {name}")

    print("\n-- Why the purity gate rejected samples --")
    gate_fails = collections.Counter(
        r["gate_reason"] for r in records if not r["fully_cutile"])
    for reason, cnt in gate_fails.most_common():
        print(f"  {cnt:5d}  {reason}")

    # The samples this stricter gate costs us: numerically correct, but not a
    # complete port. Worth calling out separately, since KernelBench's own rules
    # would have counted every one of these as a pass.
    impure_but_correct = [r for r in records
                          if r["numerically_correct"] and not r["fully_cutile"]]
    print(f"  -> {len(impure_but_correct)} samples were numerically correct but "
          f"not counted, having left PyTorch compute in place")

    halluc = collections.Counter()
    for r in records:
        halluc.update(r["hallucinated_apis"])
    if halluc:
        print("\n-- Most-invented ct.* APIs (do not exist in cuda.tile) --")
        for name, cnt in halluc.most_common(20):
            print(f"  {cnt:5d}  ct.{name}")

    solved = [pid for pid, rs in by_problem.items() if any(r["passed"] for r in rs)]
    print("\n-- Coverage --")
    print(f"  problems with >=1 passing sample: {len(solved)}/{n_problems}")


if __name__ == "__main__":
    main()
