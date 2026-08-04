"""Check that a task set is large enough for a speedup to mean anything.

A task whose reference runs in tens of microseconds is measuring kernel launch
overhead, not the kernel. Both implementations come out the same, the speedup is
noise, and training on it teaches nothing about speed -- which is exactly what
the earlier tiers do: tier 2's (2, 4, 16, 16) is 2048 elements.

So before spending sampling budget on a task set meant for the speed curriculum,
time its references and report anything below the floor.

Usage:
    python3 taskgen/audit_timing.py --level 94
    python3 taskgen/audit_timing.py --level 94 --min-ms 0.05 --prune
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--min-ms", type=float, default=0.05,
                    help="Reference time below which a task cannot be timed "
                         "meaningfully on this GPU.")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--prune", action="store_true",
                    help="Delete tasks under the floor instead of just listing.")
    args = ap.parse_args()

    import torch
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.timing import measure_ref_program_time
    from kernelbench.utils import enforce_reference_precision

    enforce_reference_precision()
    device = torch.device("cuda:0")

    dataset = construct_kernelbench_dataset(args.level)
    problems = list(dataset)
    print("timing %d references at level %d (floor %.3f ms)\n"
          % (len(problems), args.level, args.min_ms))

    timed, failed = [], []
    for problem in problems:
        try:
            stats = measure_ref_program_time(
                ref_arch_name=problem.name, ref_arch_src=problem.code,
                num_trials=args.trials, use_torch_compile=False,
                device=device, verbose=False, precision="fp32")
            if stats and "mean" in stats:
                timed.append((stats["mean"], problem.name))
            else:
                failed.append(problem.name)
        except Exception as e:
            failed.append("%s (%s)" % (problem.name, type(e).__name__))

    timed.sort()
    too_small = [(ms, n) for ms, n in timed if ms < args.min_ms]

    if timed:
        print("reference time: min %.4f ms  median %.4f ms  max %.2f ms"
              % (timed[0][0], timed[len(timed) // 2][0], timed[-1][0]))
    print("under the floor: %d/%d" % (len(too_small), len(timed)))
    for ms, name in too_small[:15]:
        print("  %8.4f ms  %s" % (ms, name))
    if failed:
        print("could not time %d: %s" % (len(failed), ", ".join(failed[:5])))

    if args.prune and too_small:
        # The dataset directory is derived the same way generate_tasks.py builds
        # it, so pruning here keeps the two in step.
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "kernelbench", "KernelBench", "level%d" % args.level)
        removed = 0
        for _, name in too_small:
            path = os.path.join(root, name)
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        print("pruned %d tasks from %s" % (removed, root))
    elif too_small:
        print("rerun with --prune to remove them")

    # A set where nothing is comfortably above the floor cannot teach speed.
    healthy = sum(1 for ms, _ in timed if ms >= args.min_ms * 4)
    print("\n%d/%d tasks are at least 4x the floor, where a speedup is solid"
          % (healthy, len(timed)))


if __name__ == "__main__":
    main()
