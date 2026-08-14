import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainScaleClampSiLU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(torch.clamp(x * 2.0, min=-1.0, max=1.0))


batch_size = 4096
dim = 2048

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
