"""Environment gate: verify cuTile compiles and runs on the target GPU.

Checks the two things the whole eval depends on: that a basic tile kernel
round-trips correctly with non-divisible sizes, and which dtypes ct.mma accepts
(KernelBench defaults to fp32, which tensor cores may not support directly).
"""

import torch
import cuda.tile as ct

print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0),
      "| cc", torch.cuda.get_device_capability(0))

TILE = 256


@ct.kernel
def vadd(a, b, c):
    i = ct.bid(0)
    x = ct.load(a, index=(i,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    y = ct.load(b, index=(i,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(c, index=(i,), tile=x + y)


n = 10000  # deliberately not divisible by TILE
a = torch.randn(n, device="cuda", dtype=torch.float32)
b = torch.randn(n, device="cuda", dtype=torch.float32)
c = torch.empty_like(a)
ct.launch(torch.cuda.current_stream(), (ct.cdiv(n, TILE), 1, 1), vadd, (a, b, c))
torch.cuda.synchronize()
print("vadd max err:", (c - (a + b)).abs().max().item())

TM = TN = TK = 64


@ct.kernel
def gemm(A, B, C):
    bx, by = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(TM, TK))):
        at = ct.load(A, index=(bx, k), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
        bt = ct.load(B, index=(k, by), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(at, bt, acc)
    ct.store(C, index=(bx, by), tile=acc)


for dt in (torch.float32, torch.bfloat16, torch.float16):
    M = K = N = 256
    A = torch.randn(M, K, device="cuda", dtype=dt)
    B = torch.randn(K, N, device="cuda", dtype=dt)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    try:
        ct.launch(torch.cuda.current_stream(),
                  (ct.cdiv(M, TM), ct.cdiv(N, TN), 1), gemm, (A, B, C))
        torch.cuda.synchronize()
        ref = A.float() @ B.float()
        print(f"mma {str(dt):18s} OK   max err {(C - ref).abs().max().item():.4g}")
    except Exception as e:
        print(f"mma {str(dt):18s} FAIL {type(e).__name__}: {str(e)[:220]}")
