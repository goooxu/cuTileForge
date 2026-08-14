#!/usr/bin/env python3
"""Build the standalone eval suite from HELDOUT2 and level 83.

Does not re-sample the builders. Copies the sealed / harvest sources, then
applies a fixed mutation (graph, hyperparameters, shapes) so the published
ruler is not byte-identical to anything a model has already been scored on.

Python 3.6: this has to run on the file-edit host, which cannot import
operators.py (dataclasses).
"""
from __future__ import print_function

import argparse
import ast
import hashlib
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Tables copied from operators.py so this file stays importable on 3.6
# ---------------------------------------------------------------------------

_KB_ACT_SHAPE = (4096, 393216)
BW_2D = [
    (4096, 327680),
    (2048, 524288),
    (8192, 163840),
    (1024, 1048576),
    (4096, 262144),
    (16384, 65536),
    (8192, 131072),
    (2048, 655360),
    (512, 2097152),
    (32768, 32768),
    (4096, 245760),
    (6144, 163840),
]
EVAL_BW_2D = [
    (3072, 327680),
    (2304, 524288),
    (7168, 163840),
    (1280, 1048576),
    (5120, 262144),
    (12288, 81920),
    (9216, 131072),
    (2560, 524288),
    (768, 1572864),
    (28672, 40960),
    (3584, 294912),
    (6656, 163840),
]
assert _KB_ACT_SHAPE not in EVAL_BW_2D
assert not (set(EVAL_BW_2D) & set(BW_2D))
assert all(0.8e9 <= m * k <= 1.4e9 for m, k in EVAL_BW_2D)
assert all(m % 256 == 0 and k % 256 == 0 for m, k in EVAL_BW_2D)

POINTWISE_EXPR = {
    "ReLU": "torch.relu(x)",
    "LeakyReLU": "torch.nn.functional.leaky_relu(x, negative_slope=0.01)",
    "Sigmoid": "torch.sigmoid(x)",
    "Tanh": "torch.tanh(x)",
    "GELU": "torch.nn.functional.gelu(x)",
    "GELUTanh": "torch.nn.functional.gelu(x, approximate='tanh')",
    "SELU": "torch.selu(x)",
    "ELU": "torch.nn.functional.elu(x, alpha=1.0)",
    "CELU": "torch.nn.functional.celu(x, alpha=1.0)",
    "Softplus": "torch.nn.functional.softplus(x)",
    "Softsign": "torch.nn.functional.softsign(x)",
    "HardSigmoid": "torch.nn.functional.hardsigmoid(x)",
    "HardSwish": "torch.nn.functional.hardswish(x)",
    "HardTanh": "torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)",
    "HardShrink": "torch.nn.functional.hardshrink(x, lambd=0.5)",
    "SoftShrink": "torch.nn.functional.softshrink(x, lambd=0.5)",
    "TanhShrink": "torch.nn.functional.tanhshrink(x)",
    "LogSigmoid": "torch.nn.functional.logsigmoid(x)",
    "SiLU": "torch.nn.functional.silu(x)",
    "Mish": "torch.nn.functional.mish(x)",
    "ReLU6": "torch.nn.functional.relu6(x)",
}
# Cycle permutation: same 21 ops, shifted by one, so speed-track counts hold.
POINTWISE_CYCLE = [
    "ReLU", "ReLU6", "LeakyReLU", "ELU", "CELU", "SELU",
    "GELU", "GELUTanh", "SiLU", "Mish", "HardSwish",
    "Sigmoid", "HardSigmoid", "LogSigmoid",
    "Tanh", "TanhShrink", "HardTanh",
    "HardShrink", "SoftShrink", "Softplus", "Softsign",
]
assert sorted(POINTWISE_CYCLE) == sorted(POINTWISE_EXPR)

FUSION_TAILS = [
    ("ReLU", "torch.relu({})"),
    ("Sigmoid", "torch.sigmoid({})"),
    ("Tanh", "torch.tanh({})"),
    ("Scale", "({}) * 2.0"),
    ("AddBias", "({}) + 1.5"),
    ("GELU", "torch.nn.functional.gelu({})"),
    ("SiLU", "torch.nn.functional.silu({})"),
    ("Mish", "torch.nn.functional.mish({})"),
    ("HardSwish", "torch.nn.functional.hardswish({})"),
    ("Softplus", "torch.nn.functional.softplus({})"),
    ("Clamp", "torch.clamp({}, min=-1.0, max=1.0)"),
    ("LeakyReLU", "torch.nn.functional.leaky_relu({}, negative_slope=0.01)"),
    ("ELU", "torch.nn.functional.elu({})"),
]

LEAF_TO_LABEL = {
    "relu": "ReLU", "relu6": "ReLU6", "leaky_relu": "LeakyReLU",
    "sigmoid": "Sigmoid", "tanh": "Tanh", "gelu": "GELU", "selu": "SELU",
    "elu": "ELU", "celu": "CELU", "softplus": "Softplus", "softsign": "Softsign",
    "hardsigmoid": "HardSigmoid", "hardswish": "HardSwish",
    "hardtanh": "HardTanh", "hardshrink": "HardShrink",
    "softshrink": "SoftShrink", "tanhshrink": "TanhShrink",
    "logsigmoid": "LogSigmoid", "silu": "SiLU", "mish": "Mish",
    "clamp": "Clamp", "abs": "Abs",
}
POINTWISE_LEAVES = set(LEAF_TO_LABEL) | {"hardswish"}
REDUCING_LEAVES = {"softmax", "log_softmax", "softmin", "logsumexp"}
BACKBONE_LEAVES = {
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "linear", "matmul", "bmm", "mm", "addmm",
    "layer_norm", "group_norm", "instance_norm", "batch_norm",
    "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
    "mse_loss", "l1_loss", "huber_loss", "smooth_l1_loss",
    "cross_entropy", "nll_loss", "kl_div",
    "binary_cross_entropy", "binary_cross_entropy_with_logits",
}

HPARAM_RULES = [
    (re.compile(r"negative_slope\s*=\s*0\.01\b"), "negative_slope=0.02"),
    (re.compile(r"(?<![A-Za-z_])alpha\s*=\s*1\.0\b"), "alpha=1.25"),
    (re.compile(r"lambd\s*=\s*0\.5\b"), "lambd=0.3"),
    (re.compile(r"min_val\s*=\s*-1\.0\b"), "min_val=-2.0"),
    (re.compile(r"max_val\s*=\s*1\.0\b"), "max_val=2.0"),
    (re.compile(r"eps:\s*float\s*=\s*1e-5\b"), "eps: float = 1e-4"),
    (re.compile(r"eps\s*=\s*1e-5\b"), "eps=1e-4"),
    (re.compile(r"(?<![A-Za-z_])beta\s*=\s*1\.0\b"), "beta=1.5"),
    (re.compile(r"(?<![A-Za-z_])delta\s*=\s*1\.0\b"), "delta=0.75"),
]
HPARAM_ALT = [
    (re.compile(r"negative_slope\s*=\s*0\.02\b"), "negative_slope=0.05"),
    (re.compile(r"(?<![A-Za-z_])alpha\s*=\s*1\.25\b"), "alpha=1.5"),
]

BATCH_NAMES = ("batch_size", "BATCH_SIZE")
SPATIAL_NAMES = (
    "height", "width", "depth", "HEIGHT", "WIDTH", "DEPTH",
    "INPUT_HEIGHT", "INPUT_WIDTH", "INPUT_DEPTH",
)
DIM_CAP = 20000000
DOC_RE = re.compile(r'"""(.+?) \(tier (\d+), (\w+)\)"""')
ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)\s*$", re.M)
SOFTMAX_RE = re.compile(r"softmax|softmin", re.I)

# torch / F / nn names that mutations are allowed to introduce.
_VOCAB_LEAVES = set(LEAF_TO_LABEL) | set(REDUCING_LEAVES) | set(BACKBONE_LEAVES)
_VOCAB_LEAVES.update(["rand", "randn", "zeros", "ones", "device", "tensor",
                      "float32", "sqrt", "pow"])


def task_hash(source):
    """Same identity as generate_tasks.task_hash (comments / blanks / spacing)."""
    body = []
    for line in source.splitlines():
        line = re.sub(r"#.*$", "", line).rstrip()
        if line.strip():
            body.append(re.sub(r"\s+", " ", line.strip()))
    return hashlib.sha256("\n".join(body).encode()).hexdigest()


def forge_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def numbered_py(directory):
    out = []
    if not os.path.isdir(directory):
        return out
    for fname in os.listdir(directory):
        if not fname.endswith(".py"):
            continue
        stem = fname.split("_", 1)[0]
        if stem.isdigit():
            out.append((int(stem), fname))
    out.sort()
    return [fname for _, fname in out]


def hashes_in_dir(directory):
    seen = set()
    if not os.path.isdir(directory):
        return seen
    for fname in os.listdir(directory):
        if fname.endswith(".py"):
            path = os.path.join(directory, fname)
            seen.add(task_hash(open(path, encoding="utf-8",
                                    errors="replace").read()))
    return seen


# ---------------------------------------------------------------------------
# Tiny unparser (3.6 has no ast.unparse / end_lineno)
# ---------------------------------------------------------------------------

def _const(node):
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Num):
        return repr(node.n)
    if isinstance(node, ast.Str):
        return repr(node.s)
    if isinstance(node, ast.NameConstant):
        return repr(node.value)
    if isinstance(node, ast.Bytes):
        return repr(node.s)
    return None


def unparse(node):
    got = _const(node)
    if got is not None:
        return got
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return "%s.%s" % (unparse(node.value), node.attr)
    if isinstance(node, ast.Call):
        func = unparse(node.func)
        args = [unparse(a) for a in node.args]
        for kw in node.keywords:
            if kw.arg is None:
                args.append("**%s" % unparse(kw.value))
            else:
                args.append("%s=%s" % (kw.arg, unparse(kw.value)))
        return "%s(%s)" % (func, ", ".join(args))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return "-%s" % unparse(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return "+%s" % unparse(node.operand)
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
               ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
        op = ops.get(type(node.op))
        if op is None:
            raise ValueError("binop %s" % type(node.op).__name__)
        return "(%s %s %s)" % (unparse(node.left), op, unparse(node.right))
    if isinstance(node, ast.Subscript):
        return "%s[%s]" % (unparse(node.value), unparse(node.slice))
    if isinstance(node, ast.Index):
        return unparse(node.value)
    if isinstance(node, ast.Slice):
        lo = unparse(node.lower) if node.lower else ""
        hi = unparse(node.upper) if node.upper else ""
        if node.step:
            return "%s:%s:%s" % (lo, hi, unparse(node.step))
        return "%s:%s" % (lo, hi)
    if isinstance(node, ast.ExtSlice):
        return ", ".join(unparse(d) for d in node.dims)
    if isinstance(node, ast.Tuple):
        inner = ", ".join(unparse(e) for e in node.elts)
        return "(%s,)" % inner if len(node.elts) == 1 else "(%s)" % inner
    if isinstance(node, ast.List):
        return "[%s]" % ", ".join(unparse(e) for e in node.elts)
    if isinstance(node, ast.Starred):
        return "*%s" % unparse(node.value)
    raise ValueError("cannot unparse %s" % type(node).__name__)


def attr_path(node):
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def leaf_of(node):
    if not isinstance(node, ast.Call):
        return None
    path = attr_path(node.func)
    if path is None:
        return None
    return path.split(".")[-1].lower()


def classify(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
        right = _const(node.right)
        if right in ("2.0", "1.5"):
            return "pointwise"
    if not isinstance(node, ast.Call):
        return None
    path = attr_path(node.func)
    if path is None:
        return "other"
    if path.startswith("self."):
        return "backbone"
    leaf = path.split(".")[-1].lower()
    if leaf in REDUCING_LEAVES:
        return "reducing"
    if leaf in ("mean", "max", "min", "sum"):
        if any(kw.arg == "dim" for kw in node.keywords):
            return "reducing"
    if leaf in BACKBONE_LEAVES or "conv" in leaf or "pool" in leaf:
        return "backbone"
    if leaf.endswith("_loss"):
        return "backbone"
    if leaf in POINTWISE_LEAVES:
        return "pointwise"
    return "other"


def current_label(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if _const(node.right) == "2.0":
            return "Scale"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        if _const(node.right) == "1.5":
            return "AddBias"
    if not isinstance(node, ast.Call):
        return None
    leaf = leaf_of(node)
    if leaf == "gelu":
        for kw in node.keywords:
            val = _const(kw.value) if kw.value is not None else None
            if kw.arg == "approximate" and val in ("'tanh'", '"tanh"'):
                return "GELUTanh"
        return "GELU"
    return LEAF_TO_LABEL.get(leaf)


def find_forward_return(tree):
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "Model"):
            continue
        for item in node.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "forward"):
                continue
            stmts = []
            for stmt in item.body:
                if isinstance(stmt, ast.Expr) and isinstance(
                        stmt.value, (ast.Str, getattr(ast, "Constant", ast.Str))):
                    continue
                stmts.append(stmt)
            if len(stmts) == 1 and isinstance(stmts[0], ast.Return):
                if stmts[0].value is not None:
                    return stmts[0].value
    return None


def first_arg(node):
    if isinstance(node, ast.Call) and node.args:
        return node.args[0]
    if isinstance(node, ast.BinOp):
        return node.left
    return None


def is_name(node):
    return isinstance(node, ast.Name)


def can_minus_one(expr):
    kind = classify(expr)
    if kind != "pointwise":
        return False
    inner = first_arg(expr)
    if inner is None or is_name(inner):
        return False
    if classify(inner) == "reducing":
        return False
    return True


def apply_template(template, inner):
    if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", inner):
        wrapped = inner
    else:
        wrapped = "(%s)" % inner
    return template.replace("x", wrapped, 1)


def next_in_cycle(label, shift):
    if label not in POINTWISE_CYCLE:
        return None
    idx = POINTWISE_CYCLE.index(label)
    return POINTWISE_CYCLE[(idx + shift) % len(POINTWISE_CYCLE)]


def find_innermost_pointwise_pair(node):
    if classify(node) == "pointwise":
        inner = first_arg(node)
        if inner is not None and classify(inner) == "pointwise":
            deeper = find_innermost_pointwise_pair(inner)
            return deeper if deeper is not None else (node, inner)
        if inner is not None and isinstance(inner, ast.Call):
            return find_innermost_pointwise_pair(inner)
    elif isinstance(node, ast.Call) and node.args:
        return find_innermost_pointwise_pair(node.args[0])
    elif isinstance(node, ast.BinOp):
        return find_innermost_pointwise_pair(node.left)
    return None


def swap_innermost_src(expr_src, expr_ast):
    pair = find_innermost_pointwise_pair(expr_ast)
    if pair is None:
        return expr_src, False
    outer, inner = pair
    ol, il = current_label(outer), current_label(inner)
    if ol not in POINTWISE_EXPR or il not in POINTWISE_EXPR:
        return expr_src, False
    core = unparse(first_arg(inner))
    old_pair = unparse(outer)
    new_pair = apply_template(POINTWISE_EXPR[il],
                              apply_template(POINTWISE_EXPR[ol], core))
    if old_pair not in expr_src:
        return expr_src, False
    return expr_src.replace(old_pair, new_pair, 1), True


def swap_root_pointwise(expr_src, expr_ast, shift):
    if classify(expr_ast) != "pointwise":
        return expr_src, None
    # Single pointwise on a name, or pointwise wrapping something: replace
    # only when the call itself is a cycle op (not Clamp / Scale / Abs).
    label = current_label(expr_ast)
    nxt = next_in_cycle(label, shift)
    if nxt is None:
        return expr_src, None
    inner = first_arg(expr_ast)
    if inner is None:
        return expr_src, None
    new_expr = apply_template(POINTWISE_EXPR[nxt], unparse(inner))
    return new_expr, nxt


def wrap_expr(expr_src, tail_index):
    label, tmpl = FUSION_TAILS[tail_index % len(FUSION_TAILS)]
    return tmpl.format(expr_src), label


def unwrap_expr(expr_ast):
    if not can_minus_one(expr_ast):
        return None
    return unparse(first_arg(expr_ast))


def replace_return_expr(source, new_expr):
    m = re.search(r"def forward\(self[^)]*\):\n", source)
    if not m:
        return None
    pos = m.end()
    rest = source[pos:]
    dm = re.match(r'\s+""".*?"""\n', rest, re.S)
    if dm:
        pos += dm.end()
        rest = source[pos:]
    rm = re.match(r"(\s+)return ", rest)
    if not rm:
        return None
    start = pos + rm.end()
    depth = 0
    i = start
    while i < len(source):
        ch = source[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "\n" and depth == 0:
            break
        i += 1
    return source[:start] + new_expr + source[i:]


def replace_doc_name(source, new_name):
    def repl(match):
        return '"""%s (tier %s, %s)"""' % (new_name, match.group(2), match.group(3))
    out, n = DOC_RE.subn(repl, source, count=1)
    return out if n else source


def extract_doc(source):
    m = DOC_RE.search(source)
    if not m:
        return "Mutated", "0", "unknown"
    return m.group(1), m.group(2), m.group(3)


def labels_in_expr(expr_ast):
    found = []

    def walk(node):
        lab = current_label(node) if classify(node) == "pointwise" else None
        if lab:
            found.append(lab)
        if isinstance(node, ast.Call):
            for a in node.args:
                walk(a)
        elif isinstance(node, ast.BinOp):
            walk(node.left)
            walk(node.right)

    walk(expr_ast)
    found.reverse()
    return found


def count_pointwise(expr_ast):
    return len(labels_in_expr(expr_ast))


def parse_return(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    try:
        return find_forward_return(tree)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hyperparameters and shapes
# ---------------------------------------------------------------------------

def apply_hparams(source, alt=False):
    changed = []
    rules = list(HPARAM_RULES)
    if alt:
        rules.extend(HPARAM_ALT)
    for cre, repl in rules:
        nxt, n = cre.subn(repl, source)
        if n:
            changed.append(repl.split("=")[0].strip())
            source = nxt
    return source, changed


def top_assigns(source):
    # Only module-level assignments before the first def get_inputs, so we
    # do not rewrite names that live inside methods.
    cut = source.find("\ndef get_inputs")
    head = source if cut < 0 else source[:cut]
    out = []
    for m in ASSIGN_RE.finditer(head):
        out.append((m.group(1), int(m.group(2)), m.start(), m.end()))
    return out


def rewrite_assign(source, name, new_val):
    cut = source.find("\ndef get_inputs")
    head = source if cut < 0 else source[:cut]
    tail = "" if cut < 0 else source[cut:]
    nxt, n = re.subn(
        r"^(%s\s*=\s*)-?\d+\s*$" % re.escape(name),
        r"\g<1>%d" % new_val,
        head, count=1, flags=re.M)
    if not n:
        return source, False
    return nxt + tail, True


def init_inputs_empty(source):
    m = re.search(r"def get_init_inputs\(\):\n(?:\s+\"\"\".*?\"\"\"\n)?(\s+return )(.+)",
                  source, re.S)
    if not m:
        return False
    return m.group(2).strip() in ("[]", "[ ]")


def scale_int(n, scale):
    v = int(round(n * scale))
    return max(1, v)


def remap_dims_speed(source):
    assigns = {n: v for n, v, _, _ in top_assigns(source)}
    if "batch_size" not in assigns or "dim" not in assigns:
        return source, "unchanged", None, None
    old = (assigns["batch_size"], assigns["dim"])
    if old in BW_2D:
        new = EVAL_BW_2D[BW_2D.index(old)]
    else:
        new = EVAL_BW_2D[abs(hash(old)) % len(EVAL_BW_2D)]
        if new == old:
            new = EVAL_BW_2D[(EVAL_BW_2D.index(new) + 1) % len(EVAL_BW_2D)]
    source, _ = rewrite_assign(source, "batch_size", new[0])
    source, _ = rewrite_assign(source, "dim", new[1])
    return source, "eval_bw", old, new


def remap_dims_correctness(source, scale=1.5):
    assigns = {n: v for n, v, _, _ in top_assigns(source)}
    if not assigns:
        return source, "unchanged", None, None
    changed = []
    old_shape = dict(assigns)
    keys = set(assigns)
    kernel = assigns.get("kernel_size", assigns.get("KERNEL_SIZE", 3))
    plain = keys <= {"batch_size", "dim"} or keys <= {"BATCH_SIZE", "DIM"}
    if plain and init_inputs_empty(source) and "dim" in assigns and "batch_size" in assigns:
        b, d = assigns["batch_size"], assigns["dim"]
        nb, nd = scale_int(b, scale), scale_int(d, scale)
        if nb * nd > DIM_CAP:
            factor = (float(DIM_CAP) / float(b * d)) ** 0.5
            nb, nd = max(1, int(b * factor)), max(1, int(d * factor))
        if (nb, nd) != (b, d):
            source, _ = rewrite_assign(source, "batch_size", nb)
            source, _ = rewrite_assign(source, "dim", nd)
            changed.extend(["batch_size", "dim"])
    else:
        for name in BATCH_NAMES:
            if name in assigns:
                nv = scale_int(assigns[name], scale)
                if nv != assigns[name]:
                    source, ok = rewrite_assign(source, name, nv)
                    if ok:
                        changed.append(name)
        for name in SPATIAL_NAMES:
            if name in assigns and assigns[name] > kernel:
                nv = scale_int(assigns[name], scale)
                if nv != assigns[name]:
                    source, ok = rewrite_assign(source, name, nv)
                    if ok:
                        changed.append(name)
    if not changed:
        return source, "unchanged", old_shape, None
    new_assigns = {n: v for n, v, _, _ in top_assigns(source)}
    return source, "scaled", old_shape, new_assigns


# ---------------------------------------------------------------------------
# Graph mutation
# ---------------------------------------------------------------------------

def mutate_graph(source, track, index, shift, depth_choice, tail_index):
    """Return (source, info) where info has graph, depth_delta, new_name extras."""
    info = {"graph": "unchanged", "depth_delta": 0, "swap": None, "wrap": None}
    expr_ast = parse_return(source)
    if expr_ast is None:
        info["reason"] = "complex_forward"
        return source, info
    try:
        expr_src = unparse(expr_ast)
    except ValueError as exc:
        info["reason"] = "unparse:%s" % exc
        return source, info

    actions = []
    # Root pointwise swap (cycle). For a chain, this swaps the outermost
    # cycle-op if it is the root; innermost pair swap is separate.
    if classify(expr_ast) == "pointwise":
        swapped, nxt = swap_root_pointwise(expr_src, expr_ast, shift)
        if nxt is not None:
            expr_src = swapped
            actions.append("swap:%s" % nxt)
            info["swap"] = nxt
            expr_ast = ast.parse(expr_src, mode="eval").body

    swapped_pair, did_pair = swap_innermost_src(expr_src, expr_ast)
    if did_pair:
        expr_src = swapped_pair
        actions.append("reorder")
        expr_ast = ast.parse(expr_src, mode="eval").body

    if track == "speed":
        expr_src, wlabel = wrap_expr(expr_src, tail_index)
        actions.append("wrap:%s" % wlabel)
        info["wrap"] = wlabel
        info["depth_delta"] = 1
    else:
        legal = ["0"]
        legal.append("+1")
        if can_minus_one(expr_ast):
            legal.append("-1")
        pick = legal[depth_choice % len(legal)]
        if pick == "+1":
            expr_src, wlabel = wrap_expr(expr_src, tail_index)
            actions.append("wrap:%s" % wlabel)
            info["wrap"] = wlabel
            info["depth_delta"] = 1
        elif pick == "-1":
            nxt = unwrap_expr(expr_ast)
            if nxt is not None:
                expr_src = nxt
                actions.append("unwrap")
                info["depth_delta"] = -1

    if not actions:
        return source, info

    new_source = replace_return_expr(source, expr_src)
    if new_source is None:
        info["reason"] = "replace_failed"
        return source, info
    try:
        ast.parse(new_source)
    except SyntaxError:
        info["reason"] = "syntax_after_graph"
        return source, info
    if track == "speed" and SOFTMAX_RE.search(expr_src):
        info["reason"] = "softmax_on_speed"
        return source, info

    new_ast = parse_return(new_source)
    name = extract_doc(source)[0]
    if new_ast is not None:
        labs = labels_in_expr(new_ast)
        if labs:
            name = labs[0] if len(labs) == 1 else "Chain" + "".join(labs)
    new_source = replace_doc_name(new_source, name)
    info["graph"] = "+".join(actions)
    info["new_name"] = name
    return new_source, info


def safe_filename(pid, name, tier):
    stem = re.sub(r"[^A-Za-z0-9_]", "", name) or "Mutated"
    return "%d_%s_t%s.py" % (pid, stem[:80], tier)


# ---------------------------------------------------------------------------
# One problem
# ---------------------------------------------------------------------------

def mutate_one(source, track, index, src_path, exclude, orig_hash):
    """Try a few fallbacks so the result is disjoint from every used level."""
    tail_index = index
    path_h = int(hashlib.sha256(src_path.encode()).hexdigest()[:8], 16)
    attempts = [
        dict(shift=1, depth=path_h % 3, scale=1.5, alt_hp=False, tail=tail_index),
        dict(shift=2, depth=path_h % 3, scale=1.5, alt_hp=False, tail=tail_index),
        dict(shift=1, depth=(path_h + 1) % 3, scale=1.5, alt_hp=False,
             tail=tail_index + 1),
        dict(shift=1, depth=path_h % 3, scale=1.25, alt_hp=False, tail=tail_index),
        dict(shift=1, depth=path_h % 3, scale=1.5, alt_hp=True, tail=tail_index),
    ]
    last = None
    last_info = None
    for att in attempts:
        text, ginfo = mutate_graph(
            source, track, index, att["shift"], att["depth"], att["tail"])
        text, hp = apply_hparams(text, alt=att["alt_hp"])
        if track == "speed":
            text, dim_how, old_sh, new_sh = remap_dims_speed(text)
        else:
            text, dim_how, old_sh, new_sh = remap_dims_correctness(
                text, scale=att["scale"])
        info = dict(ginfo)
        info["hparams"] = hp
        info["dims"] = dim_how
        info["old_shape"] = old_sh
        info["new_shape"] = new_sh
        h = task_hash(text)
        last, last_info = text, info
        if h != orig_hash and h not in exclude:
            info["hash"] = h
            return text, info
    # Last resort: a hashed marker. Comments do not count.
    text = last + "\n_EVAL_MARK = 1\n"
    last_info["dims"] = (last_info.get("dims") or "unchanged") + "+mark"
    last_info["hash"] = task_hash(text)
    return text, last_info


def collect_exclude(forge):
    seen = set()
    pairs = [
        (os.path.join(forge, "tasks", "heldout2"), (84, 88, 99)),
        (os.path.join(forge, "tasks", "bw_act"), (83,)),
        (os.path.join(forge, "tasks", "heldout"), (97, 98)),
        (os.path.join(forge, "kernelbench", "KernelBench"),
         (1, 2, 3, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96)),
    ]
    for root, levels in pairs:
        for lvl in levels:
            seen |= hashes_in_dir(os.path.join(root, "level%d" % lvl))
    return seen


def iter_sources(forge):
    """Yield (track, source_level, directory, filename) in eval order."""
    ho = os.path.join(forge, "tasks", "heldout2")
    for lvl in (99, 88, 84):
        d = os.path.join(ho, "level%d" % lvl)
        for fname in numbered_py(d):
            yield "correctness", lvl, d, fname
    bw = os.path.join(forge, "tasks", "bw_act", "level83")
    for fname in numbered_py(bw):
        yield "speed", 83, bw, fname


def build(forge, out_root):
    exclude = collect_exclude(forge)
    print("excluding %d hashes from used levels" % len(exclude))

    buckets = {"correctness": [], "speed": []}
    for track, lvl, directory, fname in iter_sources(forge):
        src_path = os.path.join(directory, fname)
        source = open(src_path, encoding="utf-8", errors="replace").read()
        buckets[track].append((lvl, fname, src_path, source))

    assert len(buckets["correctness"]) == 770, len(buckets["correctness"])
    assert len(buckets["speed"]) == 250, len(buckets["speed"])

    manifest = []
    written = {"correctness": 0, "speed": 0}
    for track, level_id in (("correctness", 60), ("speed", 61)):
        dest = os.path.join(out_root, "level%d" % level_id)
        if os.path.isdir(dest):
            for old in os.listdir(dest):
                if old.endswith(".py"):
                    os.remove(os.path.join(dest, old))
        else:
            os.makedirs(dest)
        for i, (lvl, fname, src_path, source) in enumerate(buckets[track]):
            pid = i + 1
            orig_h = task_hash(source)
            text, info = mutate_one(
                source, track, i, src_path, exclude, orig_h)
            exclude.add(info["hash"])
            name, tier, cat = extract_doc(text)
            out_name = safe_filename(pid, info.get("new_name") or name, tier)
            with open(os.path.join(dest, out_name), "w") as fh:
                fh.write(text)
            written[track] += 1
            manifest.append({
                "problem_id": pid,
                "level": level_id,
                "file": out_name,
                "track": track,
                "source_level": lvl,
                "source_file": fname,
                "graph": info.get("graph"),
                "depth_delta": info.get("depth_delta", 0),
                "swap": info.get("swap"),
                "wrap": info.get("wrap"),
                "hparams": info.get("hparams") or [],
                "dims": info.get("dims"),
                "old_shape": info.get("old_shape"),
                "new_shape": info.get("new_shape"),
                "category": cat,
                "tier": tier,
            })
        print("wrote %d problems to %s" % (written[track], dest))

    man_path = os.path.join(out_root, "manifest.json")
    with open(man_path, "w") as fh:
        json.dump({"levels": {"correctness": 60, "speed": 61},
                   "problems": manifest}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s (%d entries)" % (man_path, len(manifest)))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", default=None,
                    help="cuTileForge root. Default: parent of taskgen/.")
    ap.add_argument("--out-root", default=None,
                    help="Where to write level60/level61. Default: tasks/eval.")
    args = ap.parse_args()
    forge = args.forge or forge_root()
    out_root = args.out_root or os.path.join(forge, "tasks", "eval")
    build(forge, out_root)


if __name__ == "__main__":
    main()
