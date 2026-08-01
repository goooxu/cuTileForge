"""Golden cuTile solution for KernelBench level 1 problem 42 (MaxPool2d).

Solvability probe for the "Grid dimensions must be at most 3" failure, which
accounts for 79 level-1 samples. The input is 4D (N, C, H, W) and the model
repeatedly asked for a 4D or 5D grid. cuTile grids are capped at 3 dimensions,
but that is not an expressiveness limit: folding N and C together on the host
gives a 3D array and a 3D grid.
"""

import torch
import torch.nn as nn
import cuda.tile as ct

TH, TW = 16, 16


@ct.kernel
def maxpool2d_kernel(x, out, kernel_size, stride, padding, dilation,
                     in_h, in_w, out_h, out_w):
    nc = ct.bid(0)          # fused batch*channel index
    th = ct.bid(1)
    tw = ct.bid(2)

    acc = ct.full((1, TH, TW), -float("inf"), dtype=ct.float32)

    # Output coordinates covered by this tile.
    oh = th * TH + ct.arange(TH, dtype=ct.int32).reshape((1, TH, 1))
    ow = tw * TW + ct.arange(TW, dtype=ct.int32).reshape((1, 1, TW))

    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = oh * stride - padding + kh * dilation
            iw = ow * stride - padding + kw * dilation

            # gather needs flat indices; clamp then mask so out-of-range taps
            # read a valid address but contribute -inf.
            valid = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
            ih_c = ct.maximum(ct.minimum(ih, in_h - 1), 0)
            iw_c = ct.maximum(ct.minimum(iw, in_w - 1), 0)
            flat = (nc * in_h + ih_c) * in_w + iw_c

            vals = ct.gather(x, flat)
            acc = ct.maximum(acc, ct.where(valid, vals,
                                           ct.full((1, TH, TW), -float("inf"),
                                                   dtype=ct.float32)))

    in_range = (oh < out_h) & (ow < out_w)
    out_flat = (nc * out_h + ct.minimum(oh, out_h - 1)) * out_w + ct.minimum(ow, out_w - 1)
    ct.scatter(out, out_flat, acc, mask=in_range)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        n, c, in_h, in_w = x.shape
        k, s, p, d = self.kernel_size, self.stride, self.padding, self.dilation
        out_h = (in_h + 2 * p - d * (k - 1) - 1) // s + 1
        out_w = (in_w + 2 * p - d * (k - 1) - 1) // s + 1

        out = torch.empty((n * c, out_h, out_w), dtype=x.dtype, device=x.device)

        grid = (n * c, ct.cdiv(out_h, TH), ct.cdiv(out_w, TW))
        ct.launch(torch.cuda.current_stream(), grid, maxpool2d_kernel,
                  (x.view(-1), out.view(-1), k, s, p, d,
                   in_h, in_w, out_h, out_w))
        return out.view(n, c, out_h, out_w)
