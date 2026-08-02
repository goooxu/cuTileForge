"""Generate a catalogue of the KernelBench Level 1/2 problems used by this eval.

Pairs each problem with how the model actually did on it, so the table doubles
as a map of which operator families the model can and cannot express in cuTile.

Shapes are read by executing each problem's get_inputs() with tensor factories
redirected to the meta device, which yields real shapes without allocating the
multi-gigabyte inputs some problems declare.

Run inside the eval container from the KernelBench checkout:
    python3 scripts/build_problem_catalog.py --out docs/PROBLEMS.md
"""

import argparse
import collections
import json
import os
import re

import torch

from kernelbench.dataset import construct_kernelbench_dataset

# Longest first so "ConvTranspose2d" is not matched as "Conv".
CATEGORY_RULES = [
    ("矩阵乘",     ["matmul", "matrixmul", "bmm", "batched_matrix", "gemm", "dot",
                    "matrixvector", "matrixscalar", "linear", "innerproduct",
                    "matrixmultiplication", "tallskinny", "irregularshape",
                    "symmetric", "triangular", "diagonal"]),
    ("卷积",       ["convtranspose", "conv1d", "conv2d", "conv3d", "conv",
                    "depthwise", "pointwise", "separable"]),
    ("池化",       ["maxpool", "avgpool", "pool", "adaptive"]),
    ("归一化",     ["batchnorm", "layernorm", "groupnorm", "instancenorm",
                    "rmsnorm", "l1norm", "l2norm", "frobenius", "norm",
                    "softmax", "logsoftmax"]),
    ("激活",       ["relu", "gelu", "elu", "selu", "silu", "swish", "sigmoid",
                    "tanh", "softplus", "softsign", "hardtanh", "hardsigmoid",
                    "hardswish", "mish", "leakyrelu"]),
    ("归约/统计",  ["sum", "mean", "max", "min", "argmax", "argmin", "prod",
                    "cumsum", "cumprod", "cumulative", "reduction", "reverse",
                    "masked", "logsumexp"]),
    ("损失函数",   ["loss", "crossentropy", "kldiv", "hinge", "huber", "cosine",
                    "triplet", "margin"]),
]


def categorise(name: str) -> str:
    key = name.lower().replace("_", "").replace("-", "")
    for label, keywords in CATEGORY_RULES:
        if any(k in key for k in keywords):
            return label
    return "逐元素/其它"


def op_chain(filename: str) -> str:
    """Level 2 filenames encode the fused chain: 2_ConvTranspose2d_BiasAdd_Clamp."""
    stem = re.sub(r"^\d+_", "", filename.replace(".py", ""))
    parts = [p for p in stem.split("_") if p]
    return " + ".join(parts)


def shapes_from_ast(code: str) -> str:
    """Read shapes straight out of get_inputs() without executing torch.

    Fallback for the handful of problems whose get_inputs() trips over torch
    internals (cache setup, repeated module init) rather than anything to do
    with the shapes themselves.
    """
    import ast

    tree = ast.parse(code)
    consts: dict = {}

    def const_eval(node):
        try:
            return ast.literal_eval(node)
        except Exception:
            pass
        if isinstance(node, ast.Name):
            return consts.get(node.id)
        if isinstance(node, ast.BinOp):
            left, right = const_eval(node.left), const_eval(node.right)
            if isinstance(left, int) and isinstance(right, int):
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            val = const_eval(node.value)
            if isinstance(val, int):
                consts[node.targets[0].id] = val

    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "get_inputs"), None)
    if fn is None:
        return "-"

    factories = {"rand", "randn", "randint", "ones", "zeros", "empty", "full", "eye"}
    shapes = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr in factories):
            continue
        dims = []
        for arg in node.args:
            v = const_eval(arg)
            if isinstance(v, int):
                dims.append(v)
            elif isinstance(v, (tuple, list)) and all(isinstance(d, int) for d in v):
                dims.extend(v)
        if dims:
            shapes.append("x".join(str(d) for d in dims))
    return ", ".join(shapes) if shapes else "-"


def input_shapes(code: str) -> str:
    """Run get_inputs() with allocations redirected to the meta device."""
    factories = ("rand", "randn", "randint", "ones", "zeros", "empty", "full",
                 "arange", "eye", "tensor")
    saved = {f: getattr(torch, f) for f in factories if hasattr(torch, f)}
    saved_cuda = torch.Tensor.cuda

    def meta_wrap(fn):
        def inner(*a, **kw):
            kw.pop("device", None)
            return fn(*a, device="meta", **kw)
        return inner

    try:
        for f, fn in saved.items():
            setattr(torch, f, meta_wrap(fn))
        torch.Tensor.cuda = lambda self, *a, **kw: self

        ns: dict = {}
        exec(compile(code, "<problem>", "exec"), ns)
        inputs = ns["get_inputs"]()
        shapes = []
        for t in inputs:
            if hasattr(t, "shape"):
                shapes.append("x".join(str(d) for d in t.shape) or "scalar")
            else:
                shapes.append(type(t).__name__)
        return ", ".join(shapes) if shapes else shapes_from_ast(code)
    except Exception:
        return shapes_from_ast(code)
    finally:
        for f, fn in saved.items():
            setattr(torch, f, fn)
        torch.Tensor.cuda = saved_cuda


def load_results(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    by_problem = collections.defaultdict(list)
    for r in json.load(open(path)):
        by_problem[r["problem_id"]].append(r)
    out = {}
    for pid, rs in by_problem.items():
        n_pass = sum(r["passed"] for r in rs)
        n_correct = sum(r["numerically_correct"] for r in rs)
        speeds = [r["speedup"] for r in rs if r["passed"] and r["speedup"]]
        errs = collections.Counter(r["error_class"] for r in rs
                                   if r["error_class"]).most_common(1)
        out[pid] = {
            "passed": n_pass,
            "total": len(rs),
            "correct": n_correct,
            "best_speedup": max(speeds) if speeds else None,
            "top_error": errs[0][0] if errs else "-",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results",
                    help="Directory holding level{N}_per_sample.json.")
    ap.add_argument("--out", default="../docs/PROBLEMS.md")
    args = ap.parse_args()

    levels = {}
    for level in (1, 2):
        dataset = construct_kernelbench_dataset(level)
        res = load_results(os.path.join(args.results_dir,
                                        "level%d_per_sample.json" % level))
        rows = []
        for problem in dataset:
            pid = int(problem.name.split("_")[0])
            r = res.get(pid, {})
            rows.append({
                "id": pid,
                "name": problem.name.replace(".py", ""),
                "chain": op_chain(problem.name),
                "category": categorise(problem.name),
                "shapes": input_shapes(problem.code),
                "passed": r.get("passed"),
                "total": r.get("total"),
                "correct": r.get("correct"),
                "speedup": r.get("best_speedup"),
                "top_error": r.get("top_error", "-"),
            })
        rows.sort(key=lambda x: x["id"])
        levels[level] = rows

    out = []
    w = out.append

    w("# KernelBench Level 1 / Level 2 题目简介")
    w("")
    w("本次评测用的 200 道题，以及 Qwen3-Coder-Next 在每道题上的表现。")
    w("通过判据是「数值正确**且**完全用 cuTile 实现」，每题采样 8 次。")
    w("")
    w("- **Level 1**（100 题）：单个算子。神经网络的基本构件——矩阵乘、卷积、归一化、")
    w("  激活、归约等，每题只做一件事。")
    w("- **Level 2**（100 题）：算子融合。每题是一条算子链（如 `Conv2d + ReLU + BiasAdd`），")
    w("  融合成一个 kernel 才有性能收益。这也是模型最容易「只移植好写的部分」的地方。")
    w("")
    w("表格由 `scripts/build_problem_catalog.py` 从题目源码与评测结果生成。")
    w("")

    # Category rollup across both levels.
    w("## 按算子类别汇总")
    w("")
    w("| 类别 | Level 1 | Level 2 | 合计题数 | 至少通过一次 | 通过样本占比 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    cats = collections.defaultdict(lambda: {1: 0, 2: 0, "solved": 0, "p": 0, "t": 0})
    for level, rows in levels.items():
        for r in rows:
            c = cats[r["category"]]
            c[level] += 1
            if r["passed"]:
                c["solved"] += 1
            c["p"] += r["passed"] or 0
            c["t"] += r["total"] or 0
    for cat, c in sorted(cats.items(), key=lambda x: -(x[1][1] + x[1][2])):
        total = c[1] + c[2]
        pct = ("%.1f%%" % (c["p"] / c["t"] * 100)) if c["t"] else "-"
        w("| %s | %d | %d | %d | %d/%d | %s |"
          % (cat, c[1], c[2], total, c["solved"], total, pct))
    w("")
    w("Level 2 的题几乎都以 conv 或 gemm 起头，类别按链条里最主导的算子归。")
    w("")
    w("这张表是全篇最有指导意义的部分：模型的能力**沿算子类别断层分布**，而不是均匀地差。")
    w("逐元素类的激活函数通过率 53.8%，而卷积只有 2.8%、归一化和池化是彻底的 0。")
    w("差别不在算法难度，而在能不能套用「一个 block 管一个 tile」这个最简单的映射——")
    w("激活函数可以，卷积、池化、归一化需要处理多维索引、跨 tile 归约和边界，模型就塌了。")
    w("归一化 0/10 尤其说明问题：softmax、LayerNorm 这类算子在 cuTile 里完全写得出来")
    w("（`golden/level1_23_softmax.py` 就是可用的实现），模型只是不会。")
    w("")

    for level in (1, 2):
        rows = levels[level]
        solved = sum(1 for r in rows if r["passed"])
        w("## Level %d（%d 题，%d 题至少通过一次）" % (level, len(rows), solved))
        w("")
        if level == 2:
            w("| # | 算子链 | 输入形状 | 通过 | 数值正确 | 最好加速比 | 主要失败原因 |")
        else:
            w("| # | 题目 | 输入形状 | 通过 | 数值正确 | 最好加速比 | 主要失败原因 |")
        w("| ---: | --- | --- | ---: | ---: | ---: | --- |")
        for r in rows:
            label = r["chain"] if level == 2 else r["name"].split("_", 1)[-1]
            sp = ("%.2fx" % r["speedup"]) if r["speedup"] else "-"
            passed = ("%d/%d" % (r["passed"], r["total"])) if r["total"] else "-"
            correct = ("%d/%d" % (r["correct"], r["total"])) if r["total"] else "-"
            err = r["top_error"] if r["passed"] != r["total"] else "-"
            w("| %d | %s | %s | %s | %s | %s | %s |"
              % (r["id"], label, r["shapes"], passed, correct, sp, err))
        w("")

    text = "\n".join(out).rstrip() + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print("wrote %s (%d problems)" % (args.out, sum(len(v) for v in levels.values())))


if __name__ == "__main__":
    main()
