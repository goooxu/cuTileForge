"""Check a hand-written cuTile solution against a KernelBench problem.

Used for the solvability question: when all 8 model samples fail a problem, this
distinguishes "the model cannot write it" from "cuTile cannot express it". Uses
eval_kernel_against_ref so the pass criterion is identical to the model's.

Usage:
    python3 scripts/check_golden.py --level 1 --problem-id 23 \
        --solution golden/level1_23_softmax.py
"""

import argparse

import torch

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref
from kernelbench.utils import read_file, set_gpu_arch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--problem-id", type=int, required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--gpu-arch", nargs="+", default=["Blackwell"])
    args = ap.parse_args()

    set_gpu_arch(args.gpu_arch)

    dataset = construct_kernelbench_dataset(args.level)
    problem = dataset.get_problem_by_id(args.problem_id)

    print(f"problem : level {args.level} #{args.problem_id}  {problem.name}")
    print(f"solution: {args.solution}")

    result = eval_kernel_against_ref(
        original_model_src=problem.code,
        custom_model_src=read_file(args.solution),
        measure_performance=True,
        verbose=False,
        num_correct_trials=5,
        num_perf_trials=100,
        backend="cutile",
        precision=torch.float32,
        device=torch.device("cuda:0"),
        build_dir="/tmp/golden_cache",
    )

    print(f"compiled   : {result.compiled}")
    print(f"correctness: {result.correctness}")
    print(f"runtime    : {result.runtime} ms")
    meta = {k: v for k, v in (result.metadata or {}).items()
            if k not in ("hardware", "device")}
    for k, v in meta.items():
        print(f"  {k}: {str(v)[:400]}")

    print("\nVERDICT:", "SOLVABLE in cuTile" if result.correctness
          else "golden attempt FAILED (see error above)")


if __name__ == "__main__":
    main()
