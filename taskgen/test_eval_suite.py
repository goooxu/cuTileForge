#!/usr/bin/env python3
"""Checks on the frozen standalone eval suite (level 60).

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
    AWKWARD_NAMES, BW_2D, DIM_CAP, EVAL_AWKWARD_2D, EVAL_BW_2D, GRID_MAX,
    GRID_NAMES, POINTWISE_CYCLE, SOFTMAX_RE, THROUGHPUT_CATS, _KB_ACT_SHAPE,
    _VOCAB_LEAVES, collect_exclude, extract_doc, forge_root,
    is_throughput_eligible, parse_return, task_hash, top_assigns, unparse,
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


def assigns_of(src):
    return {n: v for n, v, _, _ in top_assigns(src)}


def grid_vals(assigns):
    return [assigns[n] for n in GRID_NAMES if n in assigns]


def awkward_vals(assigns):
    return [assigns[n] for n in AWKWARD_NAMES if n in assigns]


def forward_src(src):
    expr = parse_return(src)
    if expr is None:
        return None
    try:
        return unparse(expr)
    except ValueError:
        return None


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
    lat = [p for p in probs if p["role"] == "latency"]
    thr = [p for p in probs if p["role"] == "throughput"]
    fails += check(len(lat) == 770, "770 latency problems (%d)" % len(lat))
    fails += check(120 <= len(thr) <= 200,
                   "throughput twins in 120-200 (%d)" % len(thr))
    fails += check(man.get("n_latency") == len(lat), "manifest n_latency")
    fails += check(man.get("n_throughput") == len(thr), "manifest n_throughput")
    fails += check(man.get("levels", {}).get("suite") == 60, "suite is level 60")
    fails += check(not os.path.isdir(os.path.join(root, "level61")),
                   "level61 is gone")
    n_files = len([n for n in os.listdir(os.path.join(root, "level60"))
                   if n.endswith(".py")])
    fails += check(n_files == len(probs),
                   "level60 file count %d == manifest %d" % (n_files, len(probs)))

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

    # Latency: no billion-element 2D, every grid axis <= GRID_MAX.
    overcap = overgrid = billion = identity = bad_cd = 0
    awk_ok = 0
    awk_lat = [p for p in lat if p["shape_kind"] == "awkward"]
    for p in lat:
        src = files[(60, p["problem_id"])][1]
        if abs(p.get("depth_delta", 0)) > 1:
            bad_cd += 1
        expr = parse_return(src)
        if expr is not None and isinstance(expr, ast.Name):
            identity += 1
        assigns = assigns_of(src)
        for v in grid_vals(assigns):
            if v > GRID_MAX:
                overgrid += 1
        if set(assigns) <= {"batch_size", "dim"} and "dim" in assigns:
            prod = assigns["batch_size"] * assigns["dim"]
            if prod > DIM_CAP:
                overcap += 1
            if prod >= 800000000:
                billion += 1
        if p["shape_kind"] == "awkward":
            vals = awkward_vals(assigns)
            if vals and any(v % 2 == 1 or v % 32 != 0 for v in vals):
                awk_ok += 1
    fails += check(bad_cd == 0, "latency |depth_delta| <= 1")
    fails += check(identity == 0, "no identity / empty forwards")
    fails += check(overcap == 0, "latency 2D stays under %d elems" % DIM_CAP)
    fails += check(billion == 0, "latency has no billion-element 2D")
    fails += check(overgrid == 0, "no latency grid axis > %d" % GRID_MAX)
    fails += check(160 <= len(awk_lat) <= 192,
                   "awkward latency is a minority slice (%d)" % len(awk_lat))
    fails += check(awk_ok == len(awk_lat),
                   "every awkward latency problem has an odd/unaligned axis")
    fails += check(all(p["problem_id"] % 4 == 0 for p in awk_lat),
                   "awkward latency ids are multiples of 4")
    fails += check(all(p["graph_id"] == p["problem_id"] for p in lat),
                   "latency graph_id == problem_id")

    # Throughput twins.
    by_graph = {p["problem_id"]: p for p in lat}
    bad_parent = bad_fwd = bad_cat = 0
    common_bad = awk_bad = align_common = align_awk = 0
    kb = bw = 0
    for p in thr:
        src = files[(60, p["problem_id"])][1]
        parent = by_graph.get(p["graph_id"])
        if parent is None:
            bad_parent += 1
            continue
        psrc = files[(60, parent["problem_id"])][1]
        if forward_src(src) != forward_src(psrc):
            bad_fwd += 1
        if p["category"] not in THROUGHPUT_CATS:
            bad_cat += 1
        if not is_throughput_eligible(src, p["category"]):
            bad_cat += 1
        assigns = assigns_of(src)
        sh = (assigns.get("batch_size"), assigns.get("dim"))
        if None in sh or not (0.8e9 <= sh[0] * sh[1] <= 1.4e9):
            if p["shape_kind"] == "awkward":
                awk_bad += 1
            else:
                common_bad += 1
        if p["shape_kind"] == "awkward" and (
                sh[0] > GRID_MAX or sh[1] > GRID_MAX):
            awk_bad += 1
        if sh == _KB_ACT_SHAPE:
            kb += 1
        if sh in BW_2D:
            bw += 1
        if p["shape_kind"] == "common":
            if sh not in EVAL_BW_2D or sh[0] % 256 != 0 or sh[1] % 256 != 0:
                align_common += 1
        else:
            if sh not in EVAL_AWKWARD_2D or sh[0] % 256 == 0 or sh[1] % 256 == 0:
                align_awk += 1
        if SOFTMAX_RE.search(src.split("def get_inputs")[0]):
            bad_cat += 1
    eligible_lat = []
    for p in lat:
        src = files[(60, p["problem_id"])][1]
        if is_throughput_eligible(src, p["category"]):
            eligible_lat.append(p["problem_id"])
    twin_parents = sorted(p["graph_id"] for p in thr)
    fails += check(twin_parents == sorted(eligible_lat),
                   "every eligible latency graph has exactly one twin")
    fails += check(bad_parent == 0, "every twin points at a latency parent")
    fails += check(bad_fwd == 0, "twin forward matches parent")
    fails += check(bad_cat == 0, "twins are activation/elementwise, no softmax")
    fails += check(common_bad == 0, "common throughput is EVAL_BW_2D in 0.8e9-1.4e9")
    fails += check(awk_bad == 0, "awkward throughput is EVAL_AWKWARD_2D, both <= GRID_MAX")
    fails += check(align_common == 0, "common throughput is 256-aligned")
    fails += check(align_awk == 0, "awkward throughput is not 256-aligned")
    fails += check(kb == 0, "throughput does not use the KernelBench test shape")
    fails += check(bw == 0, "throughput does not reuse BW_2D")
    awk_thr = [p for p in thr if p["shape_kind"] == "awkward"]
    fails += check(all(p["graph_id"] % 3 == 0 for p in awk_thr),
                   "awkward throughput parents have graph_id %% 3 == 0")
    want_awk_thr = sum(1 for gid in eligible_lat if gid % 3 == 0)
    fails += check(len(awk_thr) == want_awk_thr,
                   "awkward throughput count %d == parents %% 3 == 0 (%d)"
                   % (len(awk_thr), want_awk_thr))

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
    extra -= {"rand", "randn", "zeros", "ones", "device", "tensor", "float32"}
    fails += check(not extra, "no torch names outside the source vocabulary (%s)"
                   % (", ".join(sorted(extra)[:8]) if extra else "none"))

    by_src = {}
    for p in lat:
        by_src[p["source_level"]] = by_src.get(p["source_level"], 0) + 1
    fails += check(by_src.get(99) == 400, "400 from level 99")
    fails += check(by_src.get(88) == 250, "250 from level 88")
    fails += check(by_src.get(84) == 120, "120 from level 84")

    cats = {}
    for p in lat:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    fails += check(cats.get("loss", 0) == 5,
                   "loss family still 5 problems (%s)" % cats.get("loss"))

    print("\n%s" % ("all checks passed" if not fails else "%d FAILED" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
