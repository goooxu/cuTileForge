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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("-n", type=int, default=24, help="Kernels to sample.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category", default=None,
                    help="Restrict to one category, e.g. conv.")
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

    agree = disagree = 0
    for pid, path in cands:
        with open(path) as f:
            code = f.read()
        try:
            res = eval_kernel_against_ref(
                original_model_src=refs[pid], custom_model_src=code,
                seed_num=42, num_correct_trials=2, num_perf_trials=1,
                measure_performance=False, verbose=False,
                backend="cutile", precision=dtype,
                device=__import__("torch").device("cuda:0"))
            ok = bool(res and res.correctness)
            detail = "" if ok else str((res.metadata if res else {}))[:90]
        except Exception as e:
            ok, detail = False, "%s: %s" % (type(e).__name__, str(e)[:80])

        # Every kernel here was accepted by the fast verifier, so anything the
        # harness rejects is a disagreement.
        if ok:
            agree += 1
        else:
            disagree += 1
        print("  %-28s %-8s %s" % (names[pid][:28], "AGREE" if ok else "DIFFER",
                                   detail))

    print("\n%d/%d agree with KernelBench (%.0f%%)"
          % (agree, len(cands), agree / max(len(cands), 1) * 100))
    if disagree:
        print("%d disagreements: the fast verifier is too lenient somewhere"
              % disagree)


if __name__ == "__main__":
    main()
