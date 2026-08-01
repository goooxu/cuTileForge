"""Golden cuTile solution for KernelBench level 1 problem 3 (batched matmul).

Solvability probe: 7 of 8 model samples failed here with rank errors, mostly by
mixing 2D tile shapes with a 3D array. The fix is to keep every index and shape
rank-3 and put the batch on the grid's z axis.
"""

import torch
import torch.nn as nn
import cuda.tile as ct

TM, TN, TK = 64, 64, 32


@ct.kernel
def bmm_kernel(A, B, C, n_k_tiles):
    bx, by, bz = ct.bid(0), ct.bid(1), ct.bid(2)

    # Tile rank must match array rank, so the batch axis stays as a size-1 dim
    # rather than being indexed away.
    acc = ct.zeros((1, TM, TN), dtype=ct.float32)
    for k in range(n_k_tiles):
        a = ct.load(A, index=(bz, bx, k), shape=(1, TM, TK),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(bz, k, by), shape=(1, TK, TN),
                    padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(bz, bx, by), tile=acc)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        batch, m, k = A.shape
        n = B.shape[2]
        C = torch.empty((batch, m, n), dtype=A.dtype, device=A.device)

        grid = (ct.cdiv(m, TM), ct.cdiv(n, TN), batch)
        ct.launch(torch.cuda.current_stream(), grid, bmm_kernel,
                  (A, B, C, ct.cdiv(k, TK)))
        return C
