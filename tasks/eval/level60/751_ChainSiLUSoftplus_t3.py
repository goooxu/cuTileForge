import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSiLUSoftplus (tier 3, activation)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(torch.nn.functional.silu(x))


batch_size = 6144
dim = 3072
def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return []
