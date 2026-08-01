"""Probe how cuTile's matmul handles fp32, and whether a true-fp32 path exists.

KernelBench's fp32 tolerance is 1e-4, but ct.mma on fp32 appears to route through
TF32 tensor cores (~1e-2 error). This determines whether the ~40 matmul problems
in Level 1/2 are passable at the default precision.
"""

import inspect

import torch
import cuda.tile as ct

for name in ("mma", "matmul", "mma_scaled"):
    fn = getattr(ct, name, None)
    print("=" * 70)
    print(f"### ct.{name}")
    if fn is None:
        print("  MISSING")
        continue
    try:
        print("  sig:", inspect.signature(fn))
    except Exception as e:
        print("  sig unavailable:", e)
    doc = inspect.getdoc(fn) or ""
    print("  doc:", doc[:1500])

print("=" * 70)
print("### RoundingMode members:", [m for m in dir(ct.RoundingMode) if not m.startswith("_")])
print("### has tfloat32:", hasattr(ct, "tfloat32"))

TM = TN = TK = 64
M = K = N = 256


def run(kernel, A, B, out_dtype=torch.float32):
    C = torch.zeros(M, N, device="cuda", dtype=out_dtype)
    ct.launch(torch.cuda.current_stream(),
              (ct.cdiv(M, TM), ct.cdiv(N, TN), 1), kernel, (A, B, C))
    torch.cuda.synchronize()
    return C


@ct.kernel
def gemm_mma(A, B, C):
    bx, by = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(TM, TK))):
        at = ct.load(A, index=(bx, k), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
        bt = ct.load(B, index=(k, by), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(at, bt, acc)
    ct.store(C, index=(bx, by), tile=acc)


@ct.kernel
def gemm_matmul(A, B, C):
    bx, by = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(TM, TK))):
        at = ct.load(A, index=(bx, k), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
        bt = ct.load(B, index=(k, by), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = acc + ct.matmul(at, bt)
    ct.store(C, index=(bx, by), tile=acc)


torch.manual_seed(0)
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

print("=" * 70)
print("torch allow_tf32 (matmul):", torch.backends.cuda.matmul.allow_tf32)
print("torch float32_matmul_precision:", torch.get_float32_matmul_precision())

ref = A.double() @ B.double()
torch_fp32 = (A @ B).double()
print(f"torch fp32 vs float64 ref : max err {(torch_fp32 - ref).abs().max().item():.4g}")

for label, kern in (("ct.mma", gemm_mma), ("ct.matmul", gemm_matmul)):
    try:
        C = run(kern, A, B)
        err = (C.double() - ref).abs().max().item()
        verdict = "PASS" if err < 1e-4 else "FAIL(>1e-4)"
        print(f"{label:12s} vs float64 ref : max err {err:.4g}  -> KernelBench fp32 {verdict}")
    except Exception as e:
        print(f"{label:12s} ERROR {type(e).__name__}: {str(e)[:200]}")
