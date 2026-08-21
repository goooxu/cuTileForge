import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainClampLeakyReLUSiLU (tier 7, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.silu(torch.nn.functional.leaky_relu(torch.clamp(x, min=-1.0, max=1.0), negative_slope=0.01))


batch_size = 7777
dim = 18000

def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
