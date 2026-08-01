import torch
import torch.nn as nn
import cuda.tile as ct

TILE_SIZE = 256


@ct.kernel
def add_kernel(a, b, out):
    # One tile block per TILE_SIZE-element chunk of the flattened arrays.
    i = ct.bid(0)
    # PaddingMode.ZERO zero-fills the tail tile when the length is not a
    # multiple of TILE_SIZE; out-of-bounds stores are discarded automatically.
    a_tile = ct.load(a, index=(i,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b, index=(i,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(i,), tile=a_tile + b_tile)


def cutile_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Host-side launcher: flattens the inputs and launches one block per tile."""
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)

    a_flat, b_flat, out_flat = a.view(-1), b.view(-1), out.view(-1)
    n = a_flat.numel()
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, add_kernel, (a_flat, b_flat, out_flat))
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return cutile_add(a, b)
