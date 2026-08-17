"""Unit-test code-block extraction against the failure modes seen in real runs.

The motivating case: several generations opened with an untagged fence holding a
formula, and extract_first_code() returned that formula as the kernel.
"""

import os
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

THINK_SKETCH = f"""<think>
A first sketch:

```python
class ModelNew(nn.Module):
    pass
```
</think>

Here is the kernel:

```python
{REAL_KERNEL}```
"""

THINK_CLOSER_ONLY = f"""A first sketch:

```python
class ModelNew(nn.Module):
    pass
```
</think>

Here is the kernel:

```python
{REAL_KERNEL}```
"""

GEMMA_EMPTY_CHANNEL = f"""<|channel>thought
<channel|>```python
{REAL_KERNEL}```
"""

GEMMA_THOUGHT_THEN_KERNEL = f"""<|channel>thought
A first sketch:

```python
class ModelNew(nn.Module):
    pass
```
<channel|>Here is the kernel:

```python
{REAL_KERNEL}```
"""

GEMMA_MID_THINK = f"""<|channel>thought
A first sketch:

```python
class ModelNew(nn.Module):
    pass
```
"""

QWEN_MID_THINK = f"""<think>
A first sketch:

```python
class ModelNew(nn.Module):
    pass
```
"""

PARSER_STRIPPED = f"""Here is the kernel:

```python
{REAL_KERNEL}```
"""

MUSE_EMPTY_SELF = f"""to=self<|message|><|eom|>```python
{REAL_KERNEL}```
"""

MUSE_THOUGHT_THEN_KERNEL = f"""to=self<|message|>
A first sketch:

```python
class ModelNew(nn.Module):
    pass
```
<|eom|>Here is the kernel:

```python
{REAL_KERNEL}```
"""

MUSE_MID_THINK = f"""to=self<|message|>
A first sketch:

```python
class ModelNew(nn.Module):
    pass
```
"""

CASES = [
    ("formula block before code", FORMULA_FIRST, True),
    ("plain single python block", PLAIN, True),
    ("untagged single block", UNTAGGED_ONLY, True),
    ("revised, takes last", REVISED, True),
    ("no fenced block", NO_FENCE, False),
    ("think sketch discarded", THINK_SKETCH, True),
    ("think closer-only discarded", THINK_CLOSER_ONLY, True),
    ("gemma empty channel", GEMMA_EMPTY_CHANNEL, True),
    ("gemma thought then kernel", GEMMA_THOUGHT_THEN_KERNEL, True),
    ("parser-stripped final content", PARSER_STRIPPED, True),
    ("muse empty to=self", MUSE_EMPTY_SELF, True),
    ("muse thought then kernel", MUSE_THOUGHT_THEN_KERNEL, True),
    ("bare ModelNew no fence", REAL_KERNEL, True),
]

failures = 0

os.environ["ENABLE_THINKING"] = "1"
for name, text, expect_none in (
    ("qwen mid-think", QWEN_MID_THINK, True),
    ("gemma mid-think", GEMMA_MID_THINK, True),
    ("parser-stripped with thinking on", PARSER_STRIPPED, False),
):
    got = extract_best_code(text, ["python", "cpp"])
    none = got is None
    ok = none if expect_none else (got is not None and "@ct.kernel" in got)
    failures += not ok
    print(f"[{'ok ' if ok else 'FAIL'}] {name}")
os.environ.pop("ENABLE_THINKING", None)

got = extract_best_code(MUSE_MID_THINK, ["python", "cpp"])
ok = got is None
failures += not ok
print(f"[{'ok ' if ok else 'FAIL'}] muse mid-think without ENABLE_THINKING")

for name, text, expect_kernel in CASES:
    got = extract_best_code(text, ["python", "cpp"])
    ok = (got is not None and "@ct.kernel" in got) if expect_kernel else (got is None)
    failures += not ok
    old = extract_first_code(text, ["python", "cpp"])
    old_ok = (old is not None and "@ct.kernel" in old) if expect_kernel else (old is None)
    note = "" if old_ok else "   <- extract_first_code got this wrong"
    print(f"[{'ok ' if ok else 'FAIL'}] {name:32s}{note}")

print()
print("all cases passed" if not failures else f"{failures} case(s) failed")
sys.exit(1 if failures else 0)
