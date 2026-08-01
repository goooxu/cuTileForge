"""Generate the cuTile API reference that gets injected into the eval prompt.

cuTile postdates the model's training data, so the docs-in-context condition needs
a reference that is complete enough to write real kernels from. This introspects
the installed cuda.tile package rather than scraping the docs site, so the
reference always matches the version the generated kernels are compiled against.

Run inside the eval container:
    python3 scripts/build_cutile_docs.py
"""

import argparse
import inspect
import os
import re

import cuda.tile as ct

# Sphinx doctest blocks make up most of the docstring bulk and add little for a
# reader who already gets worked examples elsewhere in the prompt.
_TRAILER = re.compile(
    r"\n\s*(?:Examples?:|\.\. testcode::|\.\. testoutput::|\.\. code-block::|\.\. note::)",
)

# Grouped so related operations read together, following the official docs'
# own sectioning (operations-load-store, -math, -reduction, ...).
GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Kernel definition and launch", (
        "kernel", "function", "launch", "bid", "num_blocks", "num_tiles", "cdiv",
    )),
    ("Loading and storing", (
        "load", "store", "gather", "scatter",
        "load_advanced_indexing", "store_advanced_indexing",
    )),
    ("Tile creation", (
        "zeros", "ones", "full", "arange", "astile",
    )),
    ("Matrix multiply", (
        "mma", "matmul", "mma_scaled",
    )),
    ("Reduction and scan", (
        "sum", "prod", "max", "min", "argmax", "argmin",
        "reduce", "scan", "cumsum", "cumprod",
    )),
    ("Shape and dtype", (
        "reshape", "transpose", "permute", "expand_dims", "broadcast_to",
        "cat", "extract", "astype", "bitcast",
    )),
    ("Elementwise math", (
        "abs", "add", "sub", "mul", "truediv", "floordiv", "mod", "negative", "pow",
        "exp", "exp2", "log", "log2", "sqrt", "rsqrt", "ceil", "floor",
        "sin", "cos", "tan", "sinh", "cosh", "tanh", "atan2",
        "maximum", "minimum", "isnan",
    )),
    ("Comparison and selection", (
        "where", "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    )),
    ("Bitwise", (
        "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
        "bitwise_lshift", "bitwise_rshift",
    )),
    ("Atomics", (
        "atomic_add", "atomic_max", "atomic_min", "atomic_and", "atomic_or",
        "atomic_xor", "atomic_cas", "atomic_xchg",
    )),
    ("Metaprogramming and debugging", (
        "static_eval", "static_iter", "static_assert", "assume_divisible_by",
        "assert_", "print", "printf", "pack_to_bytes", "unpack_from_bytes",
    )),
]

DTYPES = (
    "float64", "float32", "float16", "bfloat16", "tfloat32",
    "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "float4_e2m1fn",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_",
)


def condense(doc: str, max_chars: int) -> str:
    """Keep the summary/Args/Returns of a docstring, drop doctest trailers."""
    if not doc:
        return ""
    m = _TRAILER.search(doc)
    if m:
        doc = doc[: m.start()]
    doc = re.sub(r"\|([\w ]+?)\|", r"\1", doc)      # Sphinx substitutions
    doc = re.sub(r":py:\w+:`~?([^`]+)`", r"`\1`", doc)  # Sphinx cross-references
    doc = re.sub(r"``([^`]+)``", r"`\1`", doc)      # rST literals -> markdown
    doc = re.sub(r"\n{3,}", "\n\n", doc).strip()
    if len(doc) > max_chars:
        doc = doc[:max_chars].rsplit("\n", 1)[0].rstrip() + " ..."
    return doc


def signature_of(name: str, fn) -> str:
    try:
        return f"ct.{name}{inspect.signature(fn)}"
    except (TypeError, ValueError):
        return f"ct.{name}(...)"


def render(max_chars: int, concepts_path: str) -> str:
    out: list[str] = []
    w = out.append

    # The hand-written preamble carries the programming model itself (tile space,
    # power-of-two shapes, padding modes), which the per-op docstrings assume.
    if concepts_path and os.path.exists(concepts_path):
        with open(concepts_path) as f:
            w(f.read().rstrip())
        w("")

    w("# cuTile Python (`cuda.tile`) API reference")
    w("")
    w(f"Generated from cuda-tile {getattr(ct, '__version__', 'unknown')}. "
      "This is the complete set of public operations; anything not listed here "
      "does not exist in cuTile.")
    w("")

    documented: set[str] = set()

    for title, names in GROUPS:
        entries = []
        for name in names:
            fn = getattr(ct, name, None)
            if fn is None:
                continue
            documented.add(name)
            body = condense(inspect.getdoc(fn) or "", max_chars)
            entry = [f"#### `{signature_of(name, fn)}`"]
            if body:
                entry.append("")
                entry.append(body)
            entries.append("\n".join(entry))
        if entries:
            w(f"## {title}")
            w("")
            w("\n\n".join(entries))
            w("")

    w("## Data types")
    w("")
    present = [d for d in DTYPES if hasattr(ct, d)]
    documented.update(present)
    w("Referenced as `ct.<name>`: " + ", ".join(f"`{d}`" for d in present) + ".")
    w("")

    w("## Enums and classes")
    w("")
    for cls_name in ("PaddingMode", "RoundingMode", "MemoryOrder", "MemoryScope"):
        cls = getattr(ct, cls_name, None)
        if cls is None:
            continue
        documented.add(cls_name)
        members = [m for m in dir(cls) if not m.startswith("_") and m.isupper()]
        w(f"- `ct.{cls_name}`: " + ", ".join(f"`{m}`" for m in members))
    for cls_name in ("Tile", "Array", "TiledView", "Slice"):
        cls = getattr(ct, cls_name, None)
        if cls is None:
            continue
        documented.add(cls_name)
        members = [m for m in dir(cls) if not m.startswith("_")]
        w(f"- `ct.{cls_name}` attributes/methods: " + ", ".join(f"`{m}`" for m in members))
    w("")

    # Anything public but ungrouped is still listed, so the reference stays a
    # faithful allowlist even if the package adds operations later.
    public = {n for n in dir(ct) if not n.startswith("_")}
    leftover = sorted(
        n for n in public - documented
        if not inspect.ismodule(getattr(ct, n, None))
    )
    if leftover:
        w("## Other public names")
        w("")
        w(", ".join(f"`ct.{n}`" for n in leftover))
        w("")

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="src/kernelbench/prompts/cutile_api_reference.md")
    ap.add_argument("--max-doc-chars", type=int, default=900,
                    help="Per-operation docstring budget after condensing.")
    ap.add_argument("--concepts", default="src/kernelbench/prompts/cutile_concepts.md",
                    help="Hand-written programming-model preamble to prepend.")
    args = ap.parse_args()

    text = render(args.max_doc_chars, args.concepts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)

    print(f"wrote {args.out}: {len(text)} chars, ~{len(text) // 4} tokens, "
          f"{text.count('#### ')} operations")


if __name__ == "__main__":
    main()
