"""Confirm that disabling TF32 makes the torch fp32 reference agree with cuTile.

The NGC container ships with allow_tf32=True, which makes the *reference* the
imprecise side of KernelBench's 1e-4 comparison. Eval must pin true fp32.
"""

import torch
import cuda.tile as ct

TM = TN = TK = 64
M = K = N = 512


@ct.kernel
def gemm(A, B, C):
    bx, by = ct.bid(0), ct.bid(1)
    acc = ct.zeros((TM, TN), dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(TM, TK))):
        at = ct.load(A, index=(bx, k), shape=(TM, TK), padding_mode=ct.PaddingMode.ZERO)
        bt = ct.load(B, index=(k, by), shape=(TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(at, bt, acc)
    ct.store(C, index=(bx, by), tile=acc)


torch.manual_seed(0)
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
ct.launch(torch.cuda.current_stream(), (ct.cdiv(M, TM), ct.cdiv(N, TN), 1), gemm, (A, B, C))
torch.cuda.synchronize()

for tf32 in (True, False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    ref = A @ B
    diff = (ref - C).abs().max().item()
    ok = torch.allclose(ref, C, atol=1e-4, rtol=1e-4)
    print(f"allow_tf32={str(tf32):5s} -> |torch_ref - cutile| max {diff:.4g}  "
          f"allclose(1e-4)={ok}")

# Also check conv, the other big TF32 consumer in Level 1/2.
torch.backends.cudnn.allow_tf32 = False
print("cudnn.allow_tf32 set to False (affects conv reference precision)")
