#!/usr/bin/env python3
"""Checks on the frozen standalone eval suite (level 60 / 61).

The suite is an artefact, not regenerated in CI: this file reads what
build_eval_suite.py wrote and refuses to let it drift from the protocol.
"""
from __future__ import print_function

import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_eval_suite import (  # noqa: E402
    BW_2D, DIM_CAP, EVAL_BW_2D, POINTWISE_CYCLE, SOFTMAX_RE, _KB_ACT_SHAPE,
    _VOCAB_LEAVES, collect_exclude, count_pointwise, extract_doc, forge_root,
    parse_return, task_hash, top_assigns,
)


def check(cond, msg):
    print("  %-4s %s" % ("ok" if cond else "FAIL", msg))
    return 0 if cond else 1


def load_suite(forge):
    root = os.path.join(forge, "tasks", "eval")
    man = json.load(open(os.path.join(root, "manifest.json")))
    files = {}
    for p in man["problems"]:
        path = os.path.join(root, "level%d" % p["level"], p["file"])
        files[(p["level"], p["problem_id"])] = (
            p, open(path, encoding="utf-8", errors="replace").read())
    return man, files, root


def torch_leaves(source):
    return set(re.findall(
        r"(?:torch(?:\.nn(?:\.functional)?)?|F|nn)\.([A-Za-z_][A-Za-z0-9_]*)",
        source))


def source_vocab(forge):
    allowed = set(_VOCAB_LEAVES)
    allowed.update(n.lower() for n in POINTWISE_CYCLE)
    for rel, levels in (
        ("tasks/heldout2", (84, 88, 99)),
        ("tasks/bw_act", (83,)),
        ("tasks/heldout", (97, 98)),
    ):
        for lvl in levels:
            d = os.path.join(forge, rel, "level%d" % lvl)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.endswith(".py"):
                    src = open(os.path.join(d, fname), encoding="utf-8",
                               errors="replace").read()
                    allowed.update(x.lower() for x in torch_leaves(src))
    return allowed


def main():
    forge = forge_root()
    fails = 0
    man_path = os.path.join(forge, "tasks", "eval", "manifest.json")
    fails += check(os.path.isfile(man_path), "manifest.json exists")
    if not os.path.isfile(man_path):
        print("\n1 FAILED (suite not built)")
        return 1

    man, files, root = load_suite(forge)
    probs = man["problems"]
    c = [p for p in probs if p["track"] == "correctness"]
    s = [p for p in probs if p["track"] == "speed"]
    fails += check(len(c) == 770, "correctness track has 770 problems (%d)" % len(c))
    fails += check(len(s) == 250, "speed track has 250 problems (%d)" % len(s))
    fails += check(man["levels"]["correctness"] == 60, "correctness is level 60")
    fails += check(man["levels"]["speed"] == 61, "speed is level 61")
    fails += check(len(os.listdir(os.path.join(root, "level60"))) == 770,
                   "level60 directory has 770 files")
    fails += check(len(os.listdir(os.path.join(root, "level61"))) == 250,
                   "level61 directory has 250 files")

    hashes = []
    for p, src in files.values():
        hashes.append(task_hash(src))
        try:
            ast.parse(src)
        except SyntaxError:
            fails += check(False, "syntax %s" % p["file"])
    fails += check(len(set(hashes)) == len(hashes),
                   "no internal hash collisions")

    used = collect_exclude(forge)
    leak = sum(1 for h in hashes if h in used)
    fails += check(leak == 0, "no hash leaked from used levels (%d)" % leak)

    # Speed track.
    bad_shape = bad_kb = bad_bw = bad_sm = bad_dev = bad_depth = 0
    for p in s:
        src = files[(61, p["problem_id"])][1]
        assigns = {n: v for n, v, _, _ in top_assigns(src)}
        sh = (assigns.get("batch_size"), assigns.get("dim"))
        if None in sh or not (0.8e9 <= sh[0] * sh[1] <= 1.4e9):
            bad_shape += 1
        if sh == _KB_ACT_SHAPE:
            bad_kb += 1
        if sh in BW_2D:
            bad_bw += 1
        if sh not in EVAL_BW_2D:
            bad_shape += 1
        head = src.split("def get_inputs")[0]
        if SOFTMAX_RE.search(head):
            bad_sm += 1
        if "device" not in src:
            bad_dev += 1
        expr = parse_return(src)
        if expr is None or count_pointwise(expr) != 2:
            bad_depth += 1
        if abs(p.get("depth_delta", 0)) != 1:
            bad_depth += 1
    fails += check(bad_shape == 0, "every speed shape is EVAL_BW_2D in 0.8e9-1.4e9")
    fails += check(bad_kb == 0, "speed track does not use the KernelBench test shape")
    fails += check(bad_bw == 0, "speed track does not reuse BW_2D")
    fails += check(bad_sm == 0, "speed track has no softmax family")
    fails += check(bad_dev == 0, "speed get_inputs allocates on CUDA")
    fails += check(bad_depth == 0, "every speed problem is depth 2 (1->2 wrap)")

    # Correctness depth and no empty forward.
    bad_cd = identity = overcap = 0
    for p in c:
        src = files[(60, p["problem_id"])][1]
        if abs(p.get("depth_delta", 0)) > 1:
            bad_cd += 1
        expr = parse_return(src)
        if expr is not None and isinstance(expr, ast.Name):
            identity += 1
        assigns = {n: v for n, v, _, _ in top_assigns(src)}
        if set(assigns) <= {"batch_size", "dim"} and "dim" in assigns:
            if assigns["batch_size"] * assigns["dim"] > DIM_CAP:
                overcap += 1
    fails += check(bad_cd == 0, "correctness |depth_delta| <= 1")
    fails += check(identity == 0, "no identity / empty forwards")
    fails += check(overcap == 0, "correctness 2D activations stay under %d elems" % DIM_CAP)

    defaults = (
        "negative_slope=0.01", "alpha=1.0", "lambd=0.5",
        "min_val=-1.0", "eps=1e-5", "eps: float = 1e-5",
    )
    leftover = 0
    for _, src in files.values():
        for d in defaults:
            if d in src:
                leftover += 1
    fails += check(leftover == 0, "known default hyperparameters were remapped")

    allowed = source_vocab(forge)
    extra = set()
    for p, src in files.values():
        extra.update(x.lower() for x in torch_leaves(src) if x.lower() not in allowed)
    # rand / randn / device already in the seed vocab; drop obvious constructors.
    extra -= {"rand", "randn", "zeros", "ones", "device", "tensor", "float32"}
    fails += check(not extra, "no torch names outside the source vocabulary (%s)"
                   % (", ".join(sorted(extra)[:8]) if extra else "none"))

    # Source provenance: 400+250+120 and 250.
    by_src = {}
    for p in probs:
        by_src.setdefault((p["track"], p["source_level"]), 0)
        by_src[(p["track"], p["source_level"])] += 1
    fails += check(by_src.get(("correctness", 99)) == 400, "400 from level 99")
    fails += check(by_src.get(("correctness", 88)) == 250, "250 from level 88")
    fails += check(by_src.get(("correctness", 84)) == 120, "120 from level 84")
    fails += check(by_src.get(("speed", 83)) == 250, "250 from level 83")

    cats = {}
    for p, src in files.values():
        if p["track"] != "correctness":
            continue
        cats[extract_doc(src)[2]] = cats.get(extract_doc(src)[2], 0) + 1
    fails += check(cats.get("loss", 0) == 5,
                   "loss family still 5 problems (%s)" % cats.get("loss"))

    print("\n%s" % ("all checks passed" if not fails else "%d FAILED" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
