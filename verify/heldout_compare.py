#!/usr/bin/env python3
"""Read the sealed held-out sweep against the dev-set numbers.

The question this answers is narrow: the 200-problem dev set has been used to
choose between models for eleven rounds, so how much of the gain measured there
is selection overfitting? A single held-out number cannot say -- the synthetic
tasks are far easier than the benchmark. What can say is whether the ordering and
the relative spacing between models survive the change of task set.

  python3 verify/heldout_compare.py --run base:runs/ho_base --run K:runs/ho_K
"""
import argparse
import collections
import json
import os

# pass@4 solved counts on KernelBench Level 1+2, from results/phase11_comparison.txt.
DEV = {"base": (47, 200), "F": (79, 200), "H": (108, 200), "K": (134, 200)}


def read(path, k):
    by_task = collections.defaultdict(list)
    for line in open(path):
        rec = json.loads(line)
        by_task[int(rec["key"].split(":")[0])].append(rec)
    solved = sum(1 for v in by_task.values() if any(r.get("passed") for r in v))
    n_pass = sum(1 for v in by_task.values() for r in v if r.get("passed"))
    n_recs = sum(len(v) for v in by_task.values())
    return solved, len(by_task), n_pass, n_recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="TAG:PREFIX",
                    help="Prefix is the run directory without the _lNN suffix.")
    ap.add_argument("--levels", default="97,98")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    rows = []
    for spec in args.run:
        tag, _, prefix = spec.partition(":")
        per_level, tot_solved, tot_tasks, tot_pass, tot_recs = {}, 0, 0, 0, 0
        for lvl in levels:
            path = os.path.join("%s_l%d" % (prefix, lvl), "verified.jsonl")
            s, n, p, r = read(path, args.k)
            per_level[lvl] = (s, n, p, r)
            tot_solved += s
            tot_tasks += n
            tot_pass += p
            tot_recs += r
        rows.append((tag, per_level, tot_solved, tot_tasks, tot_pass, tot_recs))

    print("sealed held-out, k=%d, criterion: numerically correct AND entirely "
          "cuTile" % args.k)
    print()
    hdr = "  %-6s" % "model"
    for lvl in levels:
        hdr += "  level%-2d solved" % lvl
    hdr += "     total       pass@1"
    print(hdr)
    for tag, per, s, n, p, r in rows:
        line = "  %-6s" % tag
        for lvl in levels:
            ls, ln, _, _ = per[lvl]
            line += "   %3d/%-3d %5.1f%%" % (ls, ln, 100.0 * ls / ln)
        line += "   %3d/%-3d %5.1f%%" % (s, n, 100.0 * s / n)
        line += "   %5.1f%%" % (100.0 * p / r)
        print(line)

    # The comparison the exercise exists for. If the dev gains were an artefact
    # of having selected on dev, the held-out multiples would be the smaller
    # ones.
    print()
    print("  gain over base, held-out vs dev (pass@4 solved share)")
    base = next((row for row in rows if row[0] == "base"), None)
    if not base or "base" not in DEV:
        print("  (no base run; skipping)")
        return
    ho_base = base[2] / base[3]
    dev_base = DEV["base"][0] / DEV["base"][1]
    print("  %-6s  %-22s  %-22s" % ("model", "held-out", "dev 200"))
    for tag, per, s, n, p, r in rows:
        if tag == "base" or tag not in DEV:
            continue
        ho = s / n
        dev = DEV[tag][0] / DEV[tag][1]
        print("  %-6s  %5.1f%% (x%.2f)         %5.1f%% (x%.2f)"
              % (tag, 100.0 * ho, ho / ho_base, 100.0 * dev, dev / dev_base))


if __name__ == "__main__":
    main()
