"""Compare a fine-tuned run against the baseline on the held-out benchmark.

Both sides must come from the same evaluation protocol -- same prompt, same
"numerically correct AND entirely cuTile" criterion -- or the delta means
nothing. This reads two analyze_cutile_run.py outputs and reports the change,
including the per-category view, which is where the interesting movement is:
convolution, normalisation and pooling accounted for 92 of the 200 problems and
only 13 were ever solved at baseline.

Usage:
    python3 train/compare_runs.py \\
        --before results/level1_per_sample.json results/level2_per_sample.json \\
        --after  runs/tuned_l1/analysis.json runs/tuned_l2/analysis.json
"""

import argparse
import collections
import json
import re


CATEGORY_RULES = [
    ("matmul", ["matmul", "matrixmul", "bmm", "batched_matrix", "gemm", "dot",
                "matrixvector", "matrixscalar", "linear", "innerproduct",
                "matrixmultiplication", "tallskinny", "irregularshape",
                "symmetric", "triangular", "diagonal"]),
    ("conv", ["convtranspose", "conv1d", "conv2d", "conv3d", "conv",
              "depthwise", "pointwise", "separable"]),
    ("pool", ["maxpool", "avgpool", "pool", "adaptive"]),
    ("norm", ["batchnorm", "layernorm", "groupnorm", "instancenorm", "rmsnorm",
              "l1norm", "l2norm", "frobenius", "norm", "softmax", "logsoftmax"]),
    ("activation", ["relu", "gelu", "elu", "selu", "silu", "swish", "sigmoid",
                    "tanh", "softplus", "softsign", "hardtanh", "hardsigmoid",
                    "hardswish", "mish", "leakyrelu"]),
    ("reduction", ["sum", "mean", "max", "min", "argmax", "argmin", "prod",
                   "cumsum", "cumprod", "cumulative", "reduction", "reverse",
                   "masked", "logsumexp"]),
    ("loss", ["loss", "crossentropy", "kldiv", "hinge", "huber", "cosine",
              "triplet", "margin"]),
]


def categorise(name: str) -> str:
    key = name.lower().replace("_", "").replace("-", "")
    for label, keys in CATEGORY_RULES:
        if any(k in key for k in keys):
            return label
    return "elementwise/other"


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1.0 - k / i
    return 1.0 - p


def load(paths):
    recs = []
    for p in paths:
        recs += json.load(open(p))
    return recs


def summarise(recs, n_samples=8):
    by_problem = collections.defaultdict(list)
    for r in recs:
        by_problem[(r["problem"], r["problem_id"])].append(r)
    n, P = len(recs), len(by_problem)

    out = {
        "samples": n,
        "problems": P,
        "fully_cutile": sum(r["fully_cutile"] for r in recs) / n * 100,
        "numerically_correct": sum(r["numerically_correct"] for r in recs) / n * 100,
        "passed": sum(r["passed"] for r in recs) / n * 100,
        "solved_problems": sum(1 for rs in by_problem.values()
                               if any(x["passed"] for x in rs)),
    }
    for k in (1, 8):
        out["pass@%d" % k] = sum(
            pass_at_k(n_samples, sum(x["passed"] for x in rs), k)
            for rs in by_problem.values()) / P * 100

    cats = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        c = categorise(r["problem"])
        cats[c][1] += 1
        cats[c][0] += r["passed"]
    out["by_category"] = {c: (p, t) for c, (p, t) in cats.items()}

    errs = collections.Counter(r["error_class"] for r in recs if r["error_class"])
    out["errors"] = dict(errs.most_common(10))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    args = ap.parse_args()

    b = summarise(load(args.before), args.num_samples)
    a = summarise(load(args.after), args.num_samples)

    def delta(x, y, unit="pp"):
        d = y - x
        return "%+.1f%s" % (d, unit)

    print("=" * 74)
    print("held-out KernelBench Level 1+2: baseline vs fine-tuned")
    print("=" * 74)
    print("%-26s %10s %10s %10s" % ("metric", "before", "after", "delta"))
    for key, label in (("passed", "pass rate (samples)"),
                       ("pass@1", "pass@1"),
                       ("pass@8", "pass@8"),
                       ("numerically_correct", "numerically correct"),
                       ("fully_cutile", "entirely cuTile")):
        print("%-26s %9.1f%% %9.1f%% %10s"
              % (label, b[key], a[key], delta(b[key], a[key])))
    print("%-26s %10d %10d %+10d"
          % ("problems solved", b["solved_problems"], a["solved_problems"],
             a["solved_problems"] - b["solved_problems"]))

    print()
    print("-- by operator category (pass rate over samples) --")
    print("%-20s %8s %10s %10s %10s" % ("category", "problems", "before", "after", "delta"))
    for cat in sorted(set(b["by_category"]) | set(a["by_category"])):
        bp, bt = b["by_category"].get(cat, (0, 0))
        apn, at = a["by_category"].get(cat, (0, 0))
        br = bp / bt * 100 if bt else 0.0
        ar = apn / at * 100 if at else 0.0
        print("%-20s %8d %9.1f%% %9.1f%% %10s"
              % (cat, max(bt, at) // args.num_samples, br, ar, delta(br, ar)))

    print()
    print("-- error classes (count over all samples) --")
    keys = sorted(set(b["errors"]) | set(a["errors"]),
                  key=lambda k: -(b["errors"].get(k, 0)))
    print("%-26s %10s %10s %10s" % ("error", "before", "after", "delta"))
    for k in keys[:12]:
        bv, av = b["errors"].get(k, 0), a["errors"].get(k, 0)
        print("%-26s %10d %10d %+10d" % (k, bv, av, av - bv))


if __name__ == "__main__":
    main()
