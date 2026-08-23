#!/usr/bin/env python3
"""Turn diagnosed kernel_ms pairs into an ORPO jsonl.

Each row is one problem, one prompt, and two harvested assistant turns:
chosen is the lowest kernel_ms, rejected the highest, already filtered by
rl/diagnose_speed_pairs.py. Prompt composition matches eval
(cutile_concepts). Do not mix distill.

  python3 train/build_orpo_dataset.py \
      --pairs runs/glg_speed_pairs.jsonl --out runs/orpo_glg.jsonl
"""
from __future__ import print_function

import argparse
import collections
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tier", default="cutile_concepts")
    ap.add_argument("--ws", default=None)
    args = ap.parse_args()
    forge = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ws = args.ws or os.environ.get("CUTILE_WS") or os.path.dirname(forge)

    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt

    datasets = {}
    kept = []
    n_no_resp = 0
    for line in open(args.pairs):
        if not line.strip():
            continue
        p = json.loads(line)
        level = int(p["level"])
        pid = int(p["problem_id"])
        if level not in datasets:
            datasets[level] = construct_kernelbench_dataset(level)
        problem = datasets[level].get_problem_by_id(pid)
        prompt = get_custom_prompt(
            args.prompt_tier,
            ref_arch_src=problem.code,
            backend="cutile",
            option="one_shot",
            precision="fp32",
        )
        sides = {}
        missing = False
        for name in ("chosen", "rejected"):
            # Rebuild from ids. Absolute host paths in the pair list are
            # invisible inside the container, where the workspace is /ws.
            sid = int(p[name]["sample_id"])
            kdir = os.path.join(ws, "runs", "harvest_gle%d" % level)
            resp = os.path.join(
                kdir,
                "level_%d_problem_%d_sample_%d_response.txt" % (level, pid, sid))
            if not os.path.isfile(resp):
                missing = True
                break
            sides[name] = open(resp, encoding="utf-8", errors="replace").read()
        if missing:
            n_no_resp += 1
            continue
        kept.append({
            "level": level,
            "problem_id": pid,
            "category": p.get("category", "?"),
            "spread": p.get("spread"),
            "kernel_ms_chosen": p["chosen"]["kernel_ms"],
            "kernel_ms_rejected": p["rejected"]["kernel_ms"],
            "prompt_tier": args.prompt_tier,
            "prompt": prompt,
            "chosen": sides["chosen"],
            "rejected": sides["rejected"],
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    fam = collections.Counter(r["category"] for r in kept)
    gemm = fam.get("conv", 0) + fam.get("matmul", 0)
    print("wrote %d pairs to %s" % (len(kept), args.out))
    print("  dropped %d with no response" % n_no_resp)
    print("  families %s" % dict(fam))
    print("  GEMM %d (%.0f%%)" % (gemm, 100.0 * gemm / max(len(kept), 1)))
    if len(kept) < 80:
        print("too few pairs", file=sys.stderr)
        return 1
    if gemm / max(len(kept), 1) < 0.30:
        print("GEMM share too low", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
