import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSiLUScaleGELU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu((torch.nn.functional.silu(x) * 2.0))


batch_size = 6145
dim = 3073
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
