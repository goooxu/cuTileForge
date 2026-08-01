"""Unit-test code-block extraction against the failure modes seen in real runs.

The motivating case: several generations opened with an untagged fence holding a
formula, and extract_first_code() returned that formula as the kernel.
"""

import sys

from kernelbench.utils import extract_best_code, extract_first_code

REAL_KERNEL = """import torch
import torch.nn as nn
import cuda.tile as ct

@ct.kernel
def k(a, out):
    i = ct.bid(0)
    ct.store(out, index=(i,), tile=ct.load(a, index=(i,), shape=(256,)))

class ModelNew(nn.Module):
    def forward(self, a):
        return a
"""

FORMULA_FIRST = f"""The operation computes:

```
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

Here is the implementation:

```python
{REAL_KERNEL}```
"""

PLAIN = f"```python\n{REAL_KERNEL}```"

UNTAGGED_ONLY = f"```\n{REAL_KERNEL}```"

REVISED = f"""First attempt:

```python
import torch
class ModelNew(nn.Module):
    pass
```

Wait, that is wrong. Corrected:

```python
{REAL_KERNEL}```
"""

NO_FENCE = "I cannot write this kernel."

CASES = [
    ("formula block before code", FORMULA_FIRST, True),
    ("plain single python block", PLAIN, True),
    ("untagged single block", UNTAGGED_ONLY, True),
    ("revised, takes last", REVISED, True),
    ("no fenced block", NO_FENCE, False),
]

failures = 0
for name, text, expect_kernel in CASES:
    got = extract_best_code(text, ["python", "cpp"])
    ok = (got is not None and "@ct.kernel" in got) if expect_kernel else (got is None)
    failures += not ok
    old = extract_first_code(text, ["python", "cpp"])
    old_ok = (old is not None and "@ct.kernel" in old) if expect_kernel else (old is None)
    note = "" if old_ok else "   <- extract_first_code got this wrong"
    print(f"[{'ok ' if ok else 'FAIL'}] {name:28s}{note}")

print()
print("all cases passed" if not failures else f"{failures} case(s) failed")
sys.exit(1 if failures else 0)
