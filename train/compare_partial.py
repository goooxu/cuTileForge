"""Compare runs on whatever subset of the benchmark they all finished.

compare_runs.py needs analyze_cutile_run.py's output, which needs a complete
evaluation. Dev machines here are time-limited and have repeatedly expired
mid-run, so this reads eval_results.json directly, intersects the problems every
run actually evaluated, and recomputes the baseline on exactly that subset at
matching k.

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
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--baseline", required=True,
                    help="level<N>_per_sample.json from the baseline analysis.")
    ap.add_argument("--run", action="append", required=True,
                    metavar="LABEL:DIR", help="Repeatable: a run to compare.")
    ap.add_argument("--k", type=int, default=4,
                    help="Samples per problem to score, for both sides.")
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

    def load_run(run_dir):
        results = json.load(open(os.path.join(run_dir, "eval_results.json")))
        out = {}
        for pid, recs in results.items():
            for rec in recs:
                path = os.path.join(
                    run_dir, "level_%d_problem_%s_sample_%d_kernel.py"
                    % (args.level, pid, rec["sample_id"]))
                code = ""
                if os.path.exists(path):
                    code = open(path, encoding="utf-8", errors="replace").read()
                out[(int(pid), rec["sample_id"])] = (
                    bool(rec.get("correctness")) and bool(code) and pure(code))
        return out

    runs = []
    for spec in args.run:
        label, _, path = spec.rpartition(":")
        runs.append((label or os.path.basename(path), load_run(path)))

    baseline = {(r["problem_id"], r["sample_id"]): bool(r["passed"])
                for r in json.load(open(args.baseline))}

    pids = set.intersection(*[{p for p, _ in d} for _, d in runs])
    keys = [(p, s) for p in sorted(pids) for s in range(args.k)]
    keys = [k for k in keys if all(k in d for _, d in runs) and k in baseline]

    total = json.load(open(args.baseline))
    n_all = len({r["problem_id"] for r in total})
    print("level %d: %d of %d problems evaluated in every run, k=%d (%d samples)"
          % (args.level, len(pids), n_all, args.k, len(keys)))
    if len(pids) < n_all:
        print("PARTIAL -- the baseline below is recomputed on this same subset, "
              "not the published full-set figure")
    print()

    def row(label, d):
        ok = sum(1 for k in keys if d[k])
        solved = len({p for p, s in keys if d[(p, s)]})
        return ok, ok / max(len(keys), 1) * 100, solved

    b_ok, b_pct, b_solved = row("baseline", baseline)
    print("  %-18s %4d passed  %5.1f%%   %3d/%d problems"
          % ("baseline", b_ok, b_pct, b_solved, len(pids)))
    for label, d in runs:
        ok, pct, solved = row(label, d)
        print("  %-18s %4d passed  %5.1f%%   %3d/%d problems   %+5.1fpp vs baseline"
              % (label, ok, pct, solved, len(pids), pct - b_pct))


if __name__ == "__main__":
    main()
