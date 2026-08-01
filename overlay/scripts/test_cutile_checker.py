"""Unit-test the cuTile backend implementation check.

The important cases are the negatives: a ModelNew that passes correctness while
calling only torch ops must not be counted as a cuTile implementation.
"""

import sys

from kernelbench.kernel_static_checker import check_cutile_impl
from kernelbench.utils import read_file, get_package_resource_path

GOOD = read_file(get_package_resource_path("prompts/model_new_ex_add_cutile.py"))

TORCH_PASSTHROUGH = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, a, b):
        return a + b
"""

IMPORT_ONLY = """
import torch
import torch.nn as nn
import cuda.tile as ct

class ModelNew(nn.Module):
    def forward(self, a, b):
        return a + b
"""

NEVER_LAUNCHED = """
import torch
import torch.nn as nn
import cuda.tile as ct

@ct.kernel
def add_kernel(a, b, out):
    i = ct.bid(0)
    ct.store(out, index=(i,), tile=ct.load(a, index=(i,), shape=(256,)))

class ModelNew(nn.Module):
    def forward(self, a, b):
        return a + b
"""

TRITON_CODE = """
import torch, triton, triton.language as tl

@triton.jit
def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    tl.store(y_ptr, tl.load(x_ptr))
"""

ALIAS_VARIANT = """
import torch
import torch.nn as nn
import cuda.tile as cutile

@cutile.kernel
def k(a, out):
    i = cutile.bid(0)
    cutile.store(out, index=(i,), tile=cutile.load(a, index=(i,), shape=(128,)) * 2.0)

class ModelNew(nn.Module):
    def forward(self, a):
        out = torch.empty_like(a)
        cutile.launch(torch.cuda.current_stream(), (cutile.cdiv(a.numel(), 128), 1, 1),
                      k, (a.view(-1), out.view(-1)))
        return out
"""

CASES = [
    ("reference one-shot example", GOOD, False),
    ("pure torch passthrough", TORCH_PASSTHROUGH, True),
    ("imports cuda.tile but unused", IMPORT_ONLY, True),
    ("kernel defined but never launched", NEVER_LAUNCHED, True),
    ("triton code under cutile backend", TRITON_CODE, True),
    ("non-default import alias", ALIAS_VARIANT, False),
]

failures = 0
for name, code, expect_issue in CASES:
    has_issue, msg = check_cutile_impl(code)
    ok = has_issue == expect_issue
    failures += not ok
    print(f"[{'ok ' if ok else 'FAIL'}] {name:38s} flagged={has_issue!s:5s} {msg}")

print()
print("all cases passed" if not failures else f"{failures} case(s) failed")
sys.exit(1 if failures else 0)
