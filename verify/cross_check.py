"""Cross-check the fast verifier against KernelBench's own evaluator.

The fast verifier reimplements the correctness protocol for speed, and a subtle
divergence in it already invalidated one round of conclusions: it seeded the RNG
once instead of before each model construction, so any task whose module owns
learnable parameters could never match the reference. Numbers this pipeline
produces are therefore worth nothing unless they agree with the harness the
benchmark itself uses.

This samples harvested kernels and runs them through eval_kernel_against_ref.

Usage:
    python3 verify/cross_check.py --kernel-dir runs/repair_l93 --level 93 -n 24
"""

import argparse
import os
import random
import re
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("-n", type=int, default=24, help="Kernels to sample.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category", default=None,
                    help="Restrict to one category, e.g. conv.")
    ap.add_argument("--check-speed", action="store_true",
                    help="Also compare kernel timings against the harness.")
    ap.add_argument("--num-perf-trials", type=int, default=20)
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()

    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
    from kernelbench.utils import enforce_reference_precision

    # Same true-fp32 reference as the fast verifier, or the two are not
    # measuring the same thing.
    enforce_reference_precision()
    dtype = get_torch_dtype_from_string("fp32")

    dataset = construct_kernelbench_dataset(args.level)
    refs, names = {}, {}
    for pid in dataset.get_problem_ids():
        p = dataset.get_problem_by_id(pid)
        refs[pid], names[pid] = p.code, p.name

    pat = re.compile(r"level_%d_problem_(\d+)_sample_(\d+)_kernel\.py" % args.level)
    cands = []
    for f in sorted(os.listdir(args.kernel_dir)):
        m = pat.match(f)
        if not m:
            continue
        pid = int(m.group(1))
        if pid not in refs:
            continue
        if args.category:
            d = re.search(r'"""(\w+) \(tier \d+, (\w+)\)', refs[pid])
            if not d or d.group(2) != args.category:
                continue
        cands.append((pid, os.path.join(args.kernel_dir, f)))

    random.Random(args.seed).shuffle(cands)
    cands = cands[:args.n]
    print("cross-checking %d kernels against KernelBench's evaluator%s\n"
          % (len(cands), " (%s only)" % args.category if args.category else ""))

    # Run our verifier over the same sample rather than assuming it accepted
    # everything in the directory. A generation directory holds failures too, and
    # assuming otherwise reports every one of them as a disagreement.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from worker import verify_and_time

    items = [("%d:%d" % (pid, i),
              open(p, encoding="utf-8", errors="replace").read(), refs[pid])
             for i, (pid, p) in enumerate(cands)]
    ours = verify_and_time(items, workers=args.gpus, gpus=args.gpus,
                           num_perf_trials=args.num_perf_trials)

    agree = 0
    lenient = []   # we pass, the harness rejects
    strict = []    # the harness passes, we reject
    ratios = []
    for i, (pid, path) in enumerate(cands):
        code = open(path, encoding="utf-8", errors="replace").read()
        key = "%d:%d" % (pid, i)
        mine_ok = ours.get(key, {}).get("passed", False)
        try:
            res = eval_kernel_against_ref(
                original_model_src=refs[pid], custom_model_src=code,
                seed_num=42, num_correct_trials=2,
                num_perf_trials=args.num_perf_trials,
                measure_performance=args.check_speed, verbose=False,
                backend="cutile", precision=dtype,
                device=__import__("torch").device("cuda:0"))
            theirs_ok = bool(res and res.correctness)
        except Exception:
            res, theirs_ok = None, False

        # The harness has no purity gate, so it can accept a partial port that
        # we reject by design. Only compare where our purity gate also passed.
        our_stage = ours.get(key, {}).get("stage", "")
        verdict = "AGREE"
        if mine_ok and not theirs_ok:
            verdict, _ = "LENIENT", lenient.append(names[pid])
        elif theirs_ok and not mine_ok and our_stage != "purity":
            verdict, _ = "STRICT", strict.append(names[pid])
        elif mine_ok == theirs_ok or our_stage == "purity":
            agree += 1

        detail = ""
        if args.check_speed and mine_ok and theirs_ok and res is not None:
            mine = ours[key].get("kernel_ms")
            theirs = getattr(res, "runtime", None)
            if mine and theirs and theirs > 0:
                ratios.append(mine / theirs)
                detail = "kernel %.3f ms here vs %.3f ms there (%.2fx)" % (
                    mine, theirs, mine / theirs)
        elif not mine_ok:
            detail = "we reject: %s" % our_stage

        print("  %-28s %-8s %s" % (names[pid][:28], verdict, detail))

    print("\n%d/%d verdicts agree with KernelBench (%.0f%%)"
          % (agree, len(cands), agree / max(len(cands), 1) * 100))
    if lenient:
        print("%d too lenient (we pass, harness rejects): %s"
              % (len(lenient), ", ".join(n[:24] for n in lenient[:4])))
    if strict:
        print("%d too strict (harness passes, we reject on numerics): %s"
              % (len(strict), ", ".join(n[:24] for n in strict[:4])))

    if ratios:
        # Both sides call the same timing function, so the kernel measurement
        # should agree closely. Systematic drift here would mean training is
        # selecting for something the benchmark does not reward -- the same
        # class of silent divergence as the seeding bug.
        ratios.sort()
        med = ratios[len(ratios) // 2]
        worst = max(abs(1 - r) for r in ratios)
        print("\nkernel timing agreement over %d samples: median ratio %.3f, "
              "worst deviation %.0f%%" % (len(ratios), med, worst * 100))
        if abs(1 - med) > 0.15:
            print("MISMATCH: the two timings differ systematically; the training "
                  "signal and the reported speedup are not measuring the same thing")


if __name__ == "__main__":
    main()
