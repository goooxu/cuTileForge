import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainHardSwishGELUTanhSiLU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(torch.nn.functional.gelu((torch.nn.functional.hardswish(x)), approximate='tanh'))


batch_size = 768
dim = 1572864
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
