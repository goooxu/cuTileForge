"""Compare any number of runs on whatever subset of the benchmark they finished.

compare_runs.py needs analyze_cutile_run.py's output, which needs a complete
evaluation, and takes exactly one run. Dev machines here are time-limited and
have repeatedly expired mid-run, so this reads eval_results.json directly,
intersects the problems every run actually evaluated, and recomputes the
baseline on exactly that subset at matching k.

It does not classify errors -- use compare_runs.py for that.

That last part is the point. The evaluator walks problems in order, so a partial
run covers the low-numbered problems, which in Level 1 are mostly matmul and
score far above the benchmark average -- 57.8% against the full-set 12.6% in one
case. Reading a partial run against the published baseline would look like an
enormous gain that is entirely an artefact of which problems got evaluated.

The pass criterion matches the benchmark: numerically correct AND entirely
cuTile. The purity half is recomputed here from the kernel source, since
eval_results.json only records numerical correctness.

Usage:
    python3 train/compare_partial.py --level 1 \\
        --baseline results/level1_per_sample.json \\
        --run "A (from base)":runs/A_l1 --run "B (continued)":runs/B_l1
"""

import argparse
import importlib.util
import json
import os


# Same buckets compare_runs.py uses, so the two agree on what "conv" means.
CATEGORY_RULES = [
    ("conv", ["convtranspose", "conv1d", "conv2d", "conv3d", "conv",
              "depthwise", "pointwise", "separable"]),
    ("pool", ["maxpool", "avgpool", "pool", "adaptive"]),
    ("norm", ["batchnorm", "layernorm", "groupnorm", "instancenorm", "rmsnorm",
              "l1norm", "l2norm", "frobenius", "norm", "softmax", "logsoftmax"]),
    ("activation", ["relu", "gelu", "elu", "selu", "silu", "swish", "sigmoid",
                    "tanh", "softplus", "softsign", "hardtanh", "hardsigmoid",
                    "hardswish", "mish", "leakyrelu"]),
    ("loss", ["loss", "crossentropy", "kldiv", "hinge", "huber", "cosine",
              "triplet", "margin"]),
    ("matmul", ["matmul", "matrixmul", "bmm", "batched_matrix", "gemm", "dot",
                "matrixvector", "matrixscalar", "linear", "innerproduct",
                "matrixmultiplication", "tallskinny", "irregularshape",
                "symmetric", "triangular", "diagonal"]),
    ("reduction", ["sum", "mean", "max", "min", "argmax", "argmin", "prod",
                   "cumsum", "cumprod", "cumulative", "reduction", "reverse",
                   "masked", "logsumexp"]),
]


def categorise(name: str) -> str:
    low = name.lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in low for k in keys):
            return cat
    return "other"


def load_checker(repo_root):
    """Import the static checker without dragging in torch via the package."""
    path = os.path.join(repo_root, "kernelbench", "src", "kernelbench",
                        "kernel_static_checker.py")
    spec = importlib.util.spec_from_file_location("kernel_static_checker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    # Levels are scored together by default because the headline figures are
    # over all 200 problems; the lists below are positionally matched.
    ap.add_argument("--level", required=True,
                    help="Comma-separated levels, e.g. 1,2.")
    ap.add_argument("--baseline", required=True,
                    help="Comma-separated level<N>_per_sample.json, one per level.")
    ap.add_argument("--run", action="append", required=True,
                    metavar="LABEL:DIR[,DIR...]",
                    help="Repeatable: a run to compare, one directory per level.")
    ap.add_argument("--k", type=int, default=4,
                    help="Samples per problem to score, for both sides.")
    ap.add_argument("--by-category", action="store_true",
                    help="Break the comparison down by operator family.")
    ap.add_argument("--repo-root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    args = ap.parse_args()

    checker = load_checker(args.repo_root)

    def pure(code):
        for chk in (checker.check_cutile_impl, checker.check_torch_computation_ops,
                    checker.check_pytorch_wrap):
            if chk(code)[0]:
                return False
        return True

    levels = [int(x) for x in args.level.split(",")]
    baseline_files = args.baseline.split(",")
    if len(baseline_files) != len(levels):
        ap.error("--baseline needs one file per level")

    def load_run(run_dir, lvl):
        results = json.load(open(os.path.join(run_dir, "eval_results.json")))
        out = {}
        for pid, recs in results.items():
            for rec in recs:
                path = os.path.join(
                    run_dir, "level_%d_problem_%s_sample_%d_kernel.py"
                    % (lvl, pid, rec["sample_id"]))
                code = ""
                if os.path.exists(path):
                    code = open(path, encoding="utf-8", errors="replace").read()
                out[(lvl, int(pid), rec["sample_id"])] = (
                    bool(rec.get("correctness")) and bool(code) and pure(code))
        return out

    runs = []
    for spec in args.run:
        label, _, paths = spec.rpartition(":")
        dirs = paths.split(",")
        if len(dirs) != len(levels):
            ap.error("run %r needs one directory per level" % label)
        merged = {}
        for lvl, d in zip(levels, dirs):
            merged.update(load_run(d, lvl))
        runs.append((label or os.path.basename(dirs[0]), merged))

    baseline, total = {}, []
    for lvl, f in zip(levels, baseline_files):
        rows = json.load(open(f))
        total.extend((lvl, r) for r in rows)
        for r in rows:
            baseline[(lvl, r["problem_id"], r["sample_id"])] = bool(r["passed"])

    pids = set.intersection(*[{(l, p) for l, p, _ in d} for _, d in runs])
    keys = [(l, p, s) for l, p in sorted(pids) for s in range(args.k)]
    keys = [k for k in keys if all(k in d for _, d in runs) and k in baseline]

    n_all = len({(lvl, r["problem_id"]) for lvl, r in total})
    print("level %s: %d of %d problems evaluated in every run, k=%d (%d samples)"
          % (args.level, len(pids), n_all, args.k, len(keys)))
    if len(pids) < n_all:
        print("PARTIAL -- the baseline below is recomputed on this same subset, "
              "not the published full-set figure")
    print("criterion: numerically correct AND entirely cuTile\n")

    def row(d, subset=None):
        """pass@1 over samples, and the count of problems solved at least once."""
        ks = subset if subset is not None else keys
        ok = sum(1 for k in ks if d[k])
        solved = len({(l, p) for l, p, s in ks if d[(l, p, s)]})
        return ok, ok / max(len(ks), 1) * 100, solved

    n_probs = len({(l, p) for l, p, _ in keys})
    print("  %-18s %8s %8s %12s" % ("", "pass@1", "pass@%d" % args.k, "solved"))
    b_ok, b_pct, b_solved = row(baseline)
    print("  %-18s %7.1f%% %7.1f%% %8d/%d"
          % ("baseline", b_pct, b_solved / n_probs * 100, b_solved, n_probs))
    for label, d in runs:
        ok, pct, solved = row(d)
        print("  %-18s %7.1f%% %7.1f%% %8d/%d   %+5.1fpp pass@1, %+5.1fpp pass@%d"
              % (label, pct, solved / n_probs * 100, solved, n_probs,
                 pct - b_pct, (solved - b_solved) / n_probs * 100, args.k))

    if not args.by_category:
        return

    # Categories come from the baseline files, the only place holding names.
    names = {(lvl, r["problem_id"]): r["problem"] for lvl, r in total}
    bycat = {}
    for lp in pids:
        bycat.setdefault(categorise(names.get(lp, "")), []).append(lp)

    print("\n  by category -- pass@1, and problems solved\n")
    header = "  %-12s %6s %14s" % ("category", "probs", "baseline")
    for label, _ in runs:
        header += " %20s" % label[:20]
    print(header)
    for cat in sorted(bycat, key=lambda c: -len(bycat[c])):
        members = set(bycat[cat])
        subset = [k for k in keys if (k[0], k[1]) in members]
        if not subset:
            continue
        _, bp, bs = row(baseline, subset)
        line = "  %-12s %6d %8.1f%% %2d/%-2d" % (cat, len(members), bp, bs, len(members))
        for _, d in runs:
            _, pct, solved = row(d, subset)
            line += " %10.1f%% %2d/%-2d %+5.1f" % (pct, solved, len(members), pct - bp)
        print(line)


if __name__ == "__main__":
    main()
