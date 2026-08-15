import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainTanhShrinkSiLU (tier 2, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(torch.nn.functional.tanhshrink(x))


batch_size = 22271
dim = 49151
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
