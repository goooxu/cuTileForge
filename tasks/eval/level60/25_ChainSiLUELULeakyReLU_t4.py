import torch
import torch.nn as nn


class Model(nn.Module):
    """ChainSiLUELULeakyReLU (tier 4, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.leaky_relu(torch.nn.functional.elu((torch.nn.functional.silu(x)), alpha=1.25), negative_slope=0.02)


batch_size = 3072
dim = 6144
def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return []
