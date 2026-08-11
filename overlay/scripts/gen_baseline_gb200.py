"""Record PyTorch baseline times for Level 1 and 2 on this GPU.

results/timing has no GB200 entry, and speedups are meaningless without a
baseline measured on the same hardware and precision as the eval. The upstream
generate_baseline_time.py __main__ is interactive and defaults to bf16 across
three levels, so this does just the part we need.

Two baselines are worth having, and --torch-compile picks between them:

Eager is the weaker reference and flatters us most where Level 2 lives. Half the
benchmark is fusion chains, and every win we have there comes from PyTorch
materialising intermediates between pointwise ops -- which is exactly what
inductor removes. A speedup over eager is therefore not the number a user would
care about, since they would reach for torch.compile before writing a kernel.

Compiled is the honest reference for that reason, and the one to quote. It is
also slower to produce: inductor compiles per problem, so expect this to take
considerably longer than the eager pass.

Precision is pinned to true fp32 by measure_ref_program_time (see
utils.enforce_reference_precision), matching the correctness run.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.timing import measure_ref_program_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--num-trials", type=int, default=100)
    ap.add_argument("--torch-compile", action="store_true",
                    help="Time the reference under torch.compile instead of "
                         "eager. This is the reference to quote speedups against.")
    ap.add_argument("--compile-backend", default="inductor")
    ap.add_argument("--compile-options", default="default")
    ap.add_argument("--out", default="/ws/runs/baseline_gb200_torch_fp32.json")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    results: dict = {}

    for level in args.levels:
        dataset = construct_kernelbench_dataset(level)
        results[f"level{level}"] = {}
        for problem in tqdm(list(dataset), desc=f"level {level}"):
            try:
                stats = measure_ref_program_time(
                    ref_arch_name=problem.name,
                    ref_arch_src=problem.code,
                    num_trials=args.num_trials,
                    use_torch_compile=args.torch_compile,
                    torch_compile_backend=args.compile_backend,
                    torch_compile_options=args.compile_options,
                    device=device,
                    verbose=False,
                    precision=args.precision,
                )
                # measure_ref_program_time returns None for a few problems
                # (e.g. 95_CrossEntropyLoss) rather than raising.
                results[f"level{level}"][problem.name] = stats or {
                    "error": "measure_ref_program_time returned None"
                }
            except Exception as e:
                print(f"  [skip] {problem.name}: {type(e).__name__}: {str(e)[:160]}")
                results[f"level{level}"][problem.name] = {"error": str(e)[:400]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    for level in args.levels:
        entries = results[f"level{level}"]
        ok = sum(1 for v in entries.values() if isinstance(v, dict) and "mean" in v)
        print(f"level {level}: {ok}/{len(entries)} problems timed")
    mode = "torch.compile" if args.torch_compile else "eager"
    print(f"wrote {args.out} ({mode})")


if __name__ == "__main__":
    main()
