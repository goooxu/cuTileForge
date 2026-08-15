import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSiLUMishClampGELU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.gelu(torch.clamp(torch.nn.functional.mish((torch.nn.functional.silu(x))), min=-1.0, max=1.0))


batch_size = 16383
dim = 61441
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
