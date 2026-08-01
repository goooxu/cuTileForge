"""Golden cuTile solution for KernelBench level 1 problem 23 (Softmax).

Solvability probe: all 8 model samples failed this problem. Rows are 393216
elements wide, so a row does not fit in one tile and the kernel must stream the
row three times (max, sum, normalize) while carrying scalars across the loop.
"""

import torch
import torch.nn as nn
import cuda.tile as ct

ROW_TILE = 4096


@ct.kernel
def softmax_kernel(x, out, n_tiles):
    row = ct.bid(0)

    # Pass 1: row max. NEG_INF padding keeps the tail tile from contributing 0.
    m = ct.full((1, 1), -float("inf"), dtype=ct.float32)
    for t in range(n_tiles):
        tile = ct.load(x, index=(row, t), shape=(1, ROW_TILE),
                       padding_mode=ct.PaddingMode.NEG_INF)
        m = ct.maximum(m, ct.max(tile, axis=1, keepdims=True))

    # Pass 2: sum of exp, shifted by the max for numerical stability. Here the
    # tail must pad with NEG_INF too, so exp() of it contributes exactly 0.
    s = ct.zeros((1, 1), dtype=ct.float32)
    for t in range(n_tiles):
        tile = ct.load(x, index=(row, t), shape=(1, ROW_TILE),
                       padding_mode=ct.PaddingMode.NEG_INF)
        s = s + ct.sum(ct.exp(tile - m), axis=1, keepdims=True)

    # Pass 3: normalize and write back; out-of-bounds stores are discarded.
    for t in range(n_tiles):
        tile = ct.load(x, index=(row, t), shape=(1, ROW_TILE),
                       padding_mode=ct.PaddingMode.NEG_INF)
        ct.store(out, index=(row, t), tile=ct.exp(tile - m) / s)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        rows, cols = x.shape
        n_tiles = ct.cdiv(cols, ROW_TILE)
        ct.launch(torch.cuda.current_stream(), (rows, 1, 1),
                  softmax_kernel, (x, out, n_tiles))
        return out
